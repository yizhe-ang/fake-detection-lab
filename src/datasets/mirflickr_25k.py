"""MIRFLICKR-25k Dataset

- https://press.liacs.nl/mirflickr/
- M. J. Huiskes, M. S. Lew (2008). The MIR Flickr Retrieval Evaluation. ACM International Conference on Multimedia Information Retrieval (MIR'08), Vancouver, Canada
"""
import zipfile
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import toml
import torch
from src.datasets.utils import download_raw_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from pytorch_lightning import LightningDataModule

METADATA_FILENAME = Path("data/raw/mirflickr_25k/metadata.toml")
DL_DATA_DIRNAME = Path("data/downloaded/mirflickr_25k")
PROCESSED_DATA_DIRNAME = DL_DATA_DIRNAME / "mirflickr"


class MIRFLICKR_25kDataset(Dataset):
    def __init__(
        self,
        root_dir: Path = DL_DATA_DIRNAME,
        n_exif_attr: int = 80,
        patch_size: int = 128,
        batch_size: int = 32,
        iters_per_epoch: int = 5_000,
        label: str = "attr",  # One of {"attr", "img"}
    ) -> None:
        self.n_exif_attr = n_exif_attr
        self.patch_size = patch_size
        self.iters_per_epoch = iters_per_epoch
        self.label = label

        assert batch_size % 2 == 0, "Make sure `batch_size` is divisible by 2!"
        self.batch_size = batch_size

        self._prepare_data()

    def _prepare_data(self) -> None:
        """Download dataset files,
        and processes them into suitable data structures
        """
        if not PROCESSED_DATA_DIRNAME.exists():
            metadata = toml.load(METADATA_FILENAME)
            # Download dataset
            download_raw_dataset(metadata, DL_DATA_DIRNAME)

            # Process downloaded dataset
            print("Unzipping MIRFLICKR-25k...")
            zip = zipfile.ZipFile(DL_DATA_DIRNAME / metadata["filename"])
            zip.extractall(DL_DATA_DIRNAME)
            zip.close()

        self._init_exif_data()

    def _init_exif_data(self) -> None:
        # Compile EXIF information from dataset
        exif_dir = PROCESSED_DATA_DIRNAME / "meta" / "exif_raw"
        exif_paths = list(exif_dir.glob("*.txt"))

        data_dicts = []

        for p in exif_paths:
            d = {}

            idx = int(p.stem[4:])
            d["img_path"] = str(PROCESSED_DATA_DIRNAME / f"im{idx}.jpg")

            with p.open("r", errors="replace") as f:
                lines = f.readlines()

            for i in range(int(len(lines) / 2)):
                attr = lines[i * 2][1:].strip()
                value = lines[(i * 2) + 1].strip()

                d[attr] = value

            data_dicts.append(d)

        df = pd.DataFrame(data_dicts)

        self.img_paths = df["img_path"]

        # Determine EXIF attributes to predict
        # Select the attributes with the least missing values
        exif_attrs = list(
            df.drop("img_path", axis=1)
            .isnull()
            .mean(0)
            .sort_values()[: self.n_exif_attr]
            .index
        )
        self.exif_data = df[exif_attrs]
        self.exif_attrs = list(self.exif_data.columns)

        # TODO For a given EXIF attribute,
        # discard values that occur less than N times?

        # TODO Train / Val split?

    def _resize_img(self, img: torch.ByteTensor) -> torch.ByteTensor:
        """Resizes img if smaller than required patch size"""
        _, H, W = img.shape

        if H < self.patch_size or W < self.patch_size:
            return resize(img, size=self.patch_size)

        else:
            return img

    def _get_random_patch(self, img: torch.ByteTensor) -> torch.ByteTensor:
        _, H, W = img.shape
        rand_H = np.random.randint(H - self.patch_size + 1)
        rand_W = np.random.randint(W - self.patch_size + 1)

        return img[
            :, rand_H : rand_H + self.patch_size, rand_W : rand_W + self.patch_size
        ]

    def __getitem__(self, idx: int) -> Tuple[torch.ByteTensor, torch.LongTensor]:
        if self.label == "attr":
            return self._get_attr_batch()
        elif self.label == "img":
            return self._get_img_batch()

    def _get_attr_batch(self) -> Tuple[torch.ByteTensor, torch.LongTensor]:
        """Get pairs of image patches, and the EXIF values predictions

        Returns
        -------
        Tuple[torch.ByteTensor, torch.LongTensor]
            [2, batch_size, C, H, W], [batch_size, n_exif_attr]
            Range [0, 255],           One of {0, 1}
        """
        # FIXME Disable automatic batching? Or define a sampler?

        # TODO Include post-processing consistency pipeline

        n_rows, n_cols = self.exif_data.shape

        # Randomly choose an EXIF value
        exif_idx = np.random.randint(n_cols)

        # FIXME Cache EXIF values?
        exif_col = self.exif_data.iloc[:, exif_idx]
        while True:
            exif_value = np.random.choice(exif_col.unique())
            if exif_value is not np.nan:
                break

        # Get all images with / w/o that `exif_value`
        is_exif_value = exif_col == exif_value
        imgs_with_value = self.img_paths[is_exif_value]
        imgs_wo_value = self.img_paths[~is_exif_value]

        # [2, B, C, H, W]
        img_batch = torch.zeros(
            2, self.batch_size, 3, self.patch_size, self.patch_size, dtype=torch.uint8
        )
        # [2, B, n_exif_attr]
        attrs_batch = np.empty((2, self.batch_size, self.n_exif_attr), dtype=object)

        # FIXME Possible to vectorize this?
        # Create batch
        for batch_idx in range(self.batch_size):
            for pair_idx in (0, 1):
                # Create negative pairs for second half of the batch
                if batch_idx >= int(self.batch_size / 2):
                    imgs_to_sample = imgs_wo_value if pair_idx else imgs_with_value
                else:
                    imgs_to_sample = imgs_with_value

                img_sample = imgs_to_sample.sample()
                img_idx = img_sample.index.values[0]

                # Get attributes
                attrs = self.exif_data.loc[img_idx].values
                attrs_batch[pair_idx, batch_idx] = attrs

                # Get image
                img_path = img_sample.values[0]
                img = read_image(img_path)
                # Resize image if smaller than patch size
                img = self._resize_img(img)
                img_patch = self._get_random_patch(img)

                img_batch[pair_idx, batch_idx] = img_patch

        # Compute labels; by comparing the attrs of each pair
        labels_batch = attrs_batch[0] == attrs_batch[1]
        labels_batch = torch.tensor(labels_batch, dtype=torch.int64)

        return img_batch, labels_batch

    def _get_img_batch(self) -> Tuple[torch.ByteTensor, torch.FloatTensor]:
        """Get pairs of image patches,
        and prediction for whether each pair came from the same image

        Returns
        -------
        Tuple[torch.ByteTensor, torch.LongTensor]
            [2, batch_size, C, H, W], [batch_size]
            Range [0, 255],           One of {0, 1}
        """
        n_rows, n_cols = self.exif_data.shape

        # Batch contains half positive pairs, and half negative pairs
        labels_batch = torch.zeros(self.batch_size, dtype=torch.int64)
        labels_batch[:int(self.batch_size / 2)] = 1

        # [2, B, C, H, W]
        img_batch = torch.zeros(
            2, self.batch_size, 3, self.patch_size, self.patch_size, dtype=torch.uint8
        )

        # Create positive pairs
        for batch_idx in range(int(self.batch_size / 2)):
            # Choose a random image to be the current pair
            img_idx = np.random.randint(n_rows)
            img_path = self.img_paths[img_idx]

            img = read_image(img_path)
            # Resize image if smaller than patch size
            img = self._resize_img(img)

            for pair_idx in (0, 1):
                img_patch = self._get_random_patch(img)
                img_batch[pair_idx, batch_idx] = img_patch

        # Create negative pairs
        for batch_idx in range(int(self.batch_size / 2), self.batch_size):
            # Choose a random pair of images
            img_idxs = np.random.choice(np.arange(n_rows), size=(2,), replace=False)

            for pair_idx in (0, 1):
                img_path = self.img_paths[img_idxs[pair_idx]]

                img = read_image(img_path)
                # Resize image if smaller than patch size
                img = self._resize_img(img)
                img_patch = self._get_random_patch(img)

                img_batch[pair_idx, batch_idx] = img_patch

        return img_batch, labels_batch

    def __len__(self):
        # Determines how many iterations per epoch
        return self.iters_per_epoch


class MIRFLICKR_25kDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: Path = DL_DATA_DIRNAME,
        n_exif_attr: int = 80,
        patch_size: int = 128,
        batch_size: int = 32,
        iters_per_epoch: int = 5_000,
        label: str = "attr",  # One of {"attr", "img"}
        n_workers: int = 18,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.n_exif_attr = n_exif_attr
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.iters_per_epoch = iters_per_epoch
        self.label = label
        self.n_workers = n_workers
        self.pin_memory = pin_memory

    def prepare_data(self, *args, **kwargs) -> None:
        self.dataset = MIRFLICKR_25kDataset(
            root_dir=self.root_dir,
            n_exif_attr=self.n_exif_attr,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            iters_per_epoch=self.iters_per_epoch,
            label=self.label,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.exif_attrs = self.dataset.exif_attrs

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=None,  # Disable automatic batching
            num_workers=self.n_workers,
            pin_memory=self.pin_memory,
        )
