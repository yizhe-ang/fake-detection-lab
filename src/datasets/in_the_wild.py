"""In-the-Wild Image Splice Dataset

- https://minyoungg.github.io/selfconsistency/
- M. Huh, A. Liu, A. Owens, A. A. Efros, Fighting Fake News: Image Splice Detection via Learned Self-Consistency In ECCV, 2018
"""
import tarfile
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import toml
import torch
from src.datasets.utils import download_raw_dataset
from torch.utils.data import Dataset

METADATA_FILENAME = Path("data/raw/in_the_wild/metadata.toml")
DL_DATA_DIRNAME = Path("data/downloaded/in_the_wild")
PROCESSED_DATA_DIRNAME = DL_DATA_DIRNAME / "label_in_wild"


class InTheWildDataset(Dataset):
    def __init__(self, root_dir=PROCESSED_DATA_DIRNAME) -> None:
        self._prepare_data()

        root_dir = Path(root_dir)

        # Get list of all image paths
        img_dir = root_dir / "images"
        self.img_paths = list(img_dir.glob("*.jpg"))

        assert (
            len(self.img_paths) == 201
        ), "Incorrect expected number of images in dataset!"

    def _prepare_data(self) -> None:
        if not PROCESSED_DATA_DIRNAME.exists():
            metadata = toml.load(METADATA_FILENAME)
            # Download dataset
            download_raw_dataset(metadata, DL_DATA_DIRNAME)

            # Process downloaded dataset
            print("Unzipping In The Wild...")
            tar = tarfile.open(DL_DATA_DIRNAME / metadata["filename"], "r:gz")
            tar.extractall(DL_DATA_DIRNAME)
            tar.close()

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Returns
        -------
        Dict[str, Any]
            img : torch.ByteTensor
                [C, H, W], range [0, 255]
            label : int
                One of {0, 1}. No meaningful labels for this dataset (all manipulated)
            map : np.ndarray (uint8)
                [H, W], values one of {0, 1}
        """
        img_path = self.img_paths[idx]

        # Get image
        img = cv2.imread(str(img_path))[:, :, [2, 1, 0]]  # [H, W, C]
        assert img.dtype == np.uint8, "Image should be of type int!"
        assert (
            img.min() >= 0 and img.max() <= 255
        ), "Image should be bounded between [0, 255]!"

        img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]

        # Get localization map
        img_name = img_path.stem
        map_path = img_path.parent.parent / "masks" / f"{img_name}.png"
        map = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
        assert map.dtype == np.uint8, "Ground-truth should be of type int!"
        assert (
            map.min() >= 0 and map.max() <= 255
        ), "Ground-truth should be bounded between [0, 255]!"

        map[map > 0] = 1

        return {"img": img, "label": 1, "map": map}

    def __len__(self):
        return len(self.img_paths)
