from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from collections.abc import Iterable


class JustRAIGS(Dataset):
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
        labels_path: str,
        img_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        val_full: bool = True,
        test_full: bool = True,
        train_folds: Optional[list[int]] = None,
        val_folds: Optional[list[int]] = None,
        test_folds: Optional[list[int]] = None,
    ):
        super().__init__()
        assert split in ["train", "val", "test"], split
        self.val_full = val_full
        self.test_full = test_full

        self._set_fold(train_folds, val_folds, test_folds, split)

        self.task1_features = self.TASK_FEATURES[0]
        self.task2_features = self.TASK_FEATURES[1:]
        self.img_root = img_root
        self.transform = transform
        (
            self.img_path,
            self.task1_label,
            self.task2_label,
            self.label_valid,
        ) = self._get_data(labels_path, split)

    def _set_fold(
        self,
        train_folds: Optional[list[int]],
        val_folds: Optional[list[int]],
        test_folds: Optional[list[int]],
        split: str,
    ):
        if split == "train":
            assert isinstance(train_folds, Iterable), type(train_folds)
            self.train_folds = train_folds

        if split == "val":
            assert isinstance(val_folds, Iterable), type(val_folds)
            self.val_folds = val_folds

        if split == "test":
            assert isinstance(test_folds, Iterable), type(test_folds)
            self.test_folds = test_folds

    def _get_data(self, labels_path: str, split: str) -> pd.DataFrame:
        df = pd.read_csv(labels_path)
        df = df[df["fold"].isin(getattr(self, f"{split}_folds"))]

        if split == "train":
            # sampling
            df_pos = df[df["Abnormal"] == 1]
            df_neg = df[df["Abnormal"] == 0]
            df_neg = df_neg.sample(n=len(df_pos))
            df = pd.concat([df_pos, df_neg])

        elif (not self.val_full) and split == "val":
            df = df[df["Abnormal"] == 1]

        elif (not self.test_full) and split == "test":
            df = df[df["Abnormal"] == 1]

        df["image_name"] = df["filename"] + df["ext"]
        img_names = df["image_name"].values
        sub_dirs = df["subdir"].values
        img_paths = []
        label_valids = []
        for i, (img_name, sub_dir) in enumerate(zip(img_names, sub_dirs)):
            # get image path
            img_path = self._get_img_path(img_name, sub_dir, self.img_root)
            img_paths.append(img_path)

            # get label validity
            label_valid = self._get_label_valid(df.iloc[i])
            label_valids.append(label_valid)

        # get labels
        task1_labels = df[self.task1_features].values
        task2_labels = df[self.task2_features].values

        return img_paths, task1_labels, task2_labels, label_valids

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        # Image Augmentation
        img = self.transform(image=img)["image"]

        return {
            "img": img,
            "task1_label": self.task1_label[idx],
            "task2_label": self.task2_label[idx],
            "label_valid": self.label_valid[idx],
        }

    def __len__(self):
        return len(self.img_path)

    def _get_img_path(self, img_name, subfolder, img_path):
        try:
            img_path = Path(self.img_root) / str(int(subfolder)) / img_name
            if not img_path.is_file():
                raise AssertionError(f"No images found for {img_name}")

        except AssertionError as e:
            raise e

        else:
            return img_path

    def _get_label_valid(self, row):
        # if validitiy is False, loss of that position should be ignored
        label_concnesus = np.array(
            row[["G1G2 Concensus " + f for f in self.task2_features]], dtype=np.uint8
        )
        if row["Is G3 Label"]:
            label_valid = np.ones_like(label_concnesus, dtype=np.uint8)
        elif row["Abnormal"] == 0:
            label_valid = np.zeros_like(label_concnesus, dtype=np.uint8)
        else:
            label_valid = label_concnesus

        return label_valid
