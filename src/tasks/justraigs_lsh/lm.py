import warnings
from typing import Any, Dict, Optional

import torch
from torch.optim import lr_scheduler as lr_scheduler
from torch.optim.optimizer import Optimizer as Optimizer
import torchmetrics as tm
from lightning import pytorch as pl
from torch import nn
from src.tasks.justraigs_lsh.loss import (
    MultiLabelFocalLoss,
)

import torch.nn.functional as F

# 경고메세지 끄기
warnings.filterwarnings(action="ignore")

MAX_UINT8 = 255


class Task2JustRAIGSLM(pl.LightningModule):
    TASK_FEATURES = [
        "Abnormal",
        "ANRS",
        "ANRI",
        "RNFLDS",
        "RNFLDI",
        "BCLVS",
        "BCLVI",
        "NVT",
        "DH",
        "LD",
        "LC",
    ]

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        scheduling: bool = False,
        focal_loss: bool = False,
        label_smoothing: float = 0.0,
        state_dict_path: Optional[str] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["net"], logger=False)
        self.net = None
        self._set_net(net)
        self.criterion = self._set_criterion()
        self.metrics = self._set_metrics()

    def _set_net(self, net: nn.Module):
        self.net = net
        if self.hparams.state_dict_path is not None:
            self.load_state_dict(torch.load(self.hparams.state_dict_path)["state_dict"])

    def _set_criterion(self):
        if self.hparams.focal_loss:
            return MultiLabelFocalLoss(reduction="none")
        else:
            return nn.BCEWithLogitsLoss(reduction="none")

    def _set_metrics(self):
        metrics = torch.nn.ModuleDict({})
        for split in ["train", "val", "test"]:
            # default metrics for multilabel classification
            # Accurcay, Specificity, Sensitivity
            _m = {
                f"{split}_auroc": tm.AUROC(
                    task="multilabel",
                    num_labels=self.net.num_classes,
                    average="macro",
                    ignore_index=MAX_UINT8,
                ),
            }
            if self.net.num_classes == 10:
                # hamming distance for task2
                _m.update(
                    {
                        f"{split}_hd": tm.MeanMetric(),
                    }
                )

            if split == "val":
                _m.update(
                    {
                        f"{split}_roc": tm.ROC(
                            task="binary",
                        )
                    }
                )

                self.val_hd_per_label = []
                self.val_valid_label = []
                self.epoch0_val_hd_per_label = None

            if split == "test":
                # if self.metric_th is None:
                _m.update(
                    {
                        f"{split}_roc": tm.ROC(
                            task="binary",
                        )
                    }
                )

                self.test_hd_per_label = []
                self.test_valid_label = []
                self.epoch0_test_hd_per_label = None

            metrics.update(_m)

        return metrics

    def forward(self, x):
        return self.net(x)

    def _shared_step(self, batch, split="train"):
        x, y, label_valid = batch["img"], batch["task2_label"], batch["label_valid"]
        batch_size = x.size(0)
        y_hat = self(x)

        loss_task2 = self._compute_loss(y_hat, y.float(), label_valid)
        loss_auxiliary = F.binary_cross_entropy(
            (F.sigmoid(y_hat) * label_valid).max(dim=1).values,
            batch["task1_label"].to(y_hat.dtype),
        )
        loss = (loss_task2 + 0.2 * loss_auxiliary) / 2

        self.log(
            f"{split}_loss_task2",
            loss_task2,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{split}_loss_auxiliary",
            loss_auxiliary,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        y_hat = y_hat.detach()

        metric_keys = [n for n in self.metrics.keys() if split in n]
        for _m in metric_keys:
            if "hd" in _m:
                # th = self.metric_th or 0.5
                th = 0.5
                sliced_y_hat = (torch.sigmoid(y_hat) > th).long()
                sliced_y_hat = sliced_y_hat[batch["task1_label"]]
                if len(sliced_y_hat) == 0:
                    continue

                _metric = self.hamming_loss(
                    y[batch["task1_label"]],
                    sliced_y_hat,
                    label_valid[batch["task1_label"]],
                )
                self.metrics[_m].update(_metric)

                continue

            elif "_roc" in _m:
                y_ignored = y

            else:
                y_ignored = y.clone()
                y_ignored[label_valid == 0] = MAX_UINT8

            self.metrics[_m].update(y_hat, y_ignored)

        if split in ["test", "val"]:
            hd_per_l, valid_l = self.hamming_loss_per_label(
                y[batch["task1_label"]],
                sliced_y_hat,
                label_valid[batch["task1_label"]],
            )
            getattr(self, f"{split}_hd_per_label").append(hd_per_l)
            getattr(self, f"{split}_valid_label").append(valid_l)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "current_lr",
            torch.tensor(current_lr, dtype=torch.float32),
            rank_zero_only=True,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self._metric_logging_on_epoch_end("train")
        self._reset_metrics("train")

    def validation_step(self, batch, batch_idx) -> None:
        self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._metric_logging_on_epoch_end("val")
        self._logging_hd_per_label("val")
        self._reset_metrics("val")

    def _logging_hd_per_label(self, split):
        if split == "val":
            hd_per_label = self.all_gather(getattr(self, f"{split}_hd_per_label"))
            valid_label = self.all_gather(getattr(self, f"{split}_valid_label"))
            # sum across all steps
            hd_per_label = sum(hd_per_label)
            valid_label = sum(valid_label)
            # sum across all gpus
            hd_per_label = hd_per_label.sum(dim=0)
            valid_label = valid_label.sum(dim=0)

            hd_per_label = hd_per_label / valid_label
            if getattr(self, f"epoch0_{split}_hd_per_label") is None:
                setattr(self, f"epoch0_{split}_hd_per_label", hd_per_label)

            hd_per_label /= getattr(self, f"epoch0_{split}_hd_per_label")
            interest_label = self.TASK_FEATURES[1:]
            for idx, l in enumerate(interest_label):
                self.log(
                    f"{split}_norm_hd_{l}",
                    hd_per_label[idx],
                    prog_bar=False,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )

            self.log(
                f"macro_{split}_hd",
                hd_per_label.sum(),
                prog_bar=True,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        # if self.metric_th is None:
        #     self._logging_th_by_youden_index("test")
        # else:
        self._logging_th_by_youden_index("test")
        self._metric_logging_on_epoch_end("test")
        self._logging_hd_per_label("test")

        self._reset_metrics("test")

    def _reset_metrics(self, key: str):
        metric_keys = [n for n in self.metrics.keys() if key in n]
        for _m in metric_keys:
            self.metrics[_m].reset()

    def _logging_th_by_youden_index(self, split):
        fpr, tpr, threshold = self.metrics[f"{split}_roc"].compute()
        youden_index = tpr - fpr
        idx = torch.argmax(youden_index)
        self.log(
            f"{split}_th-by-youden-index",
            threshold[idx],
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

    def _metric_logging_on_epoch_end(self, key: str):
        assert key in ["train", "test", "val"], key
        metric_keys = [n for n in self.metrics.keys() if key in n]
        for _m in metric_keys:
            if "_roc" in _m:
                continue

            _metric = self.metrics[_m].compute()
            self.log(
                _m,
                _metric,
                prog_bar=True,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            torch.set_float32_matmul_precision("high")
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if (self.hparams.scheduler is not None) and (self.hparams.scheduling):
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def hamming_loss(self, true_labels, predicted_labels, label_valid, eps=1e-9):
        masked_disargree = torch.not_equal(true_labels, predicted_labels) * label_valid
        Hamming_distance = torch.sum(masked_disargree, dim=1)
        if not torch.count_nonzero(label_valid):
            return torch.tensor(0, dtype=Hamming_distance.dtype)

        loss = Hamming_distance / (label_valid.sum(dim=1))
        return loss

    def hamming_loss_per_label(self, true_labels, predicted_labels, label_valid):
        masked_disargree = torch.not_equal(true_labels, predicted_labels) * label_valid
        Hamming_distance = torch.sum(masked_disargree, dim=0)
        return Hamming_distance, label_valid.sum(dim=0)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def on_train_epoch_start(self) -> None:
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step(epoch=self.current_epoch)

    def _compute_loss(self, y_hat, y, label_valid=None):
        if self.hparams.label_smoothing > 0:
            y = (
                y * (1 - self.hparams.label_smoothing)
                + self.hparams.label_smoothing / self.net.num_classes
            )

        if label_valid is not None:
            loss = self._compute_masked_loss(y_hat, y, label_valid)
        else:
            loss = self.criterion(y_hat, y)

        return loss.mean()

    def _compute_masked_loss(self, y_hat, y, label_valid):
        return self.criterion(y_hat, y) * label_valid


class Task1JustRAIGSLM(Task2JustRAIGSLM):
    def __init__(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        scheduler: Any = None,
        compile: bool = False,
        scheduling: bool = False,
        focal_loss: bool = False,
        label_smoothing: float = 0,
        state_dict_path: str | None = None,
    ):
        super().__init__(
            net,
            optimizer,
            scheduler,
            compile,
            scheduling,
            focal_loss,
            label_smoothing,
            state_dict_path,
        )

    def _shared_step(self, batch, split="train"):
        x = batch["img"]
        batch_size = x.size(0)
        y_hat = self(x)

        loss = self._compute_loss(y_hat, batch["task1_label"].float().unsqueeze(-1))
        self.log(
            f"{split}_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )

        y_hat = y_hat.detach()
        metric_keys = [n for n in self.metrics.keys() if split in n]
        for _m in metric_keys:
            self.metrics[_m].update(y_hat, batch["task1_label"].unsqueeze(-1))

        return loss

    def _set_metrics(self):
        metrics = torch.nn.ModuleDict({})
        for split in ["train", "val", "test"]:
            _m = {}
            _m.update(
                {
                    f"{split}_roc": tm.ROC(task="binary"),
                    f"{split}_acc": tm.Accuracy(task="binary"),
                    f"{split}_auroc": tm.AUROC(task="binary"),
                }
            )
            metrics.update(_m)

        return metrics

    def on_validation_epoch_end(self) -> None:
        self._metric_logging_on_epoch_end("val")
        self._logging_sen_at_spe95("val")
        self._reset_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._logging_th_by_youden_index("test")
        self._metric_logging_on_epoch_end("test")
        self._logging_sen_at_spe95("test")

        self._reset_metrics("test")

    def _logging_sen_at_spe95(self, split: str):
        fpr, tpr, _ = self.metrics[f"{split}_roc"].compute()
        desired_specificity = 0.95
        idx = torch.argmax((fpr >= (1 - desired_specificity)).to(torch.uint8))
        if (
            len(torch.unique(tpr)) == 2
            and (
                torch.unique(tpr)
                == torch.tensor([0.0, 1.0], dtype=tpr.dtype, device=tpr.device)
            ).all()
        ):
            sensitivity_at_desired_specificity = torch.tensor(
                0.0, dtype=tpr.dtype, device=tpr.device
            )

        else:
            sensitivity_at_desired_specificity = tpr[idx]

        self.log(
            f"{split}_sen@spe95",
            sensitivity_at_desired_specificity,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
            on_step=False,
            on_epoch=True,
        )
