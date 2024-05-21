from typing import Any, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader, ConcatDataset
from lightning import pytorch as pl

from src.tasks.justraigs_lsh.dataset import JustRAIGS
import torch
import numpy as np
import pprint


class JustRAIGSDM(pl.LightningDataModule):
    STATS = {
        "whole": {
            "mean": [0.3686, 0.2350, 0.1520],
            "std": [0.2785, 0.1917, 0.1561],
        },
        "crop": {
            "mean": [0.5412, 0.3380, 0.2069],
            "std": [0.2481, 0.1813, 0.1608],
        },
    }
    DEFAULT_TRANSFORM_TRAIN = [
        # MinMaxNormalization(always_apply=True),
        A.HorizontalFlip(p=0.5),
    ]
    # DEFAULT_TRANSFORM_TEST = [MinMaxNormalization(always_apply=True)]
    DEFAULT_TRANSFORM_TEST = []

    def __init__(
        self,
        labels_path: str = "/data1/DATASET",
        img_root: str = "/data1/DATASET",
        num_workers: int = 0,
        pin_memory: bool = False,
        train_batch_size: int = 8,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        input_size: Optional[int] = None,
        input_type: str = "crop",
        val_full: bool = True,
        test_full: bool = True,
        train_folds: Optional[list[int]] = [1, 2, 3],
        val_folds: Optional[list[int]] = [0],
        test_folds: Optional[list[int]] = None,
        strong_aug: bool = False,
    ):
        super().__init__()
        assert input_type in ["whole", "crop"]
        # default attributes
        self.save_hyperparameters(logger=False)
        self.transform_train = None
        self.transform_test = None
        # configure transforms
        self._configure_transforms()

        print(f"\timg_root: {img_root}")
        print(f"\tlabels path: {labels_path}")
        print(f"\tinput size: {input_size}")
        print(f"\ttrain folds: {train_folds}")
        print(f"\tval folds: {val_folds}")
        print(f"\ttest folds: {test_folds}")

    def setup(self, stage=None):
        pass

    def setup_dataset(
        self,
        split: str,
        val_full: bool = True,
        test_full: bool = True,
    ):
        datasets = [
            JustRAIGS(
                labels_path=self.hparams.labels_path,
                img_root=self.hparams.img_root,
                split=split,
                transform=(
                    self.transform_train if split == "train" else self.transform_test
                ),
                val_full=val_full,
                test_full=test_full,
                train_folds=self.hparams.train_folds,
                val_folds=self.hparams.val_folds,
                test_folds=self.hparams.test_folds,
            )
        ]
        return ConcatDataset(datasets)

    def train_dataloader(
        self,
    ) -> DataLoader[Any]:
        data_train = self.setup_dataset("train")
        return DataLoader(
            dataset=data_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=_collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        data_val = self.setup_dataset("val", val_full=self.hparams.val_full)
        return DataLoader(
            dataset=data_val,
            batch_size=(
                self.hparams.val_batch_size
                if self.hparams.val_batch_size is not None
                else self.hparams.train_batch_size
            ),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=_collate_fn,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        data_test = self.setup_dataset("test", test_full=self.hparams.test_full)
        return DataLoader(
            dataset=data_test,
            batch_size=(
                self.hparams.test_batch_size
                if self.hparams.test_batch_size is not None
                else self.hparams.train_batch_size
            ),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=_collate_fn,
            shuffle=False,
        )

    def _configure_transforms(self):
        transform_train = self.DEFAULT_TRANSFORM_TRAIN
        transform_test = self.DEFAULT_TRANSFORM_TEST

        if self.hparams.strong_aug:
            transform_train += [
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=0.3),
                        A.HueSaturationValue(p=0.3),
                        A.RGBShift(p=0.3),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(p=0.3),
                        A.GaussianBlur(p=0.3),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(p=0.3),
                        A.GridDistortion(p=0.3),
                        A.OpticalDistortion(p=0.3),
                        A.GridDropout(p=0.3),
                        A.CoarseDropout(p=0.3),
                    ],
                    p=0.1,
                ),
            ]

        # Resize or Crop
        if self.hparams.input_size is not None:
            _resize_train = [
                A.Resize(
                    self.hparams.input_size,
                    self.hparams.input_size,
                    always_apply=True,
                )
            ]
            _resize_train.append(
                A.RandomResizedCrop(
                    self.hparams.input_size,
                    self.hparams.input_size,
                    scale=(0.666, 1),
                    ratio=(1, 1),
                    interpolation=1,
                    always_apply=True,
                )
            )
            _resize_train = A.OneOf(_resize_train, p=1.0)

            _resize_test = A.Resize(
                self.hparams.input_size, self.hparams.input_size, always_apply=True
            )

            transform_train.append(_resize_train)
            transform_test.append(_resize_test)

        # Standardization
        transform_train.append(
            A.Normalize(
                mean=self.STATS[self.hparams.input_type]["mean"],
                std=self.STATS[self.hparams.input_type]["std"],
                max_pixel_value=255.0,
            )
        )
        transform_test.append(
            A.Normalize(
                mean=self.STATS[self.hparams.input_type]["mean"],
                std=self.STATS[self.hparams.input_type]["std"],
                max_pixel_value=255.0,
            )
        )

        # Totensor
        transform_train.append(ToTensorV2())
        transform_test.append(ToTensorV2())

        # print transforms
        print()
        print("transform_train:")
        pprint.pprint(transform_train)
        print("transform_test:")
        pprint.pprint(transform_test)
        print()

        self.transform_train = A.Compose(transform_train)
        self.transform_test = A.Compose(transform_test)

        self.DEFAULT_TRANSFORM_TRAIN = None
        self.DEFAULT_TRANSFORM_TEST = None


def _collate_fn(batch):
    img_list = [item["img"] for item in batch]
    task1_label_list = [item["task1_label"] for item in batch]
    task2_label_list = [item["task2_label"] for item in batch]
    label_valid = [item["label_valid"] for item in batch]
    batch = {
        "img": torch.stack(img_list),
        "task1_label": torch.tensor(np.array(task1_label_list, dtype=np.uint8)),
        "task2_label": torch.tensor(np.array(task2_label_list, dtype=np.uint8)),
        "label_valid": torch.tensor(np.array(label_valid), dtype=torch.float32),
    }
    return batch
