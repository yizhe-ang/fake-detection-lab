import hashlib
from pathlib import Path
from typing import Dict, Union
from urllib.request import urlretrieve

import gdown
from tqdm import tqdm


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def compute_sha256(filename: Union[Path, str]) -> str:
    """Return SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def download_url(url: str, filename: Path) -> None:
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec


def check_and_download_url(
    dl_dirname: Path, sha256: str, filename: str, url: str, gdrive: bool = False
) -> None:
    # If already exists, don't have to download
    filename = dl_dirname / filename
    if filename.exists():
        return filename

    # Download file
    print(f"Downloading raw dataset from {url} to {filename}...")
    if gdrive:
        gdown.download(url, str(filename), quiet=False)
    else:
        download_url(url, filename)

    # Compute and check SHA256
    print("Computing SHA-256...")

    sha256_check = compute_sha256(filename)
    if sha256_check != sha256:
        raise ValueError(
            f"Downloaded data file SHA-256 ({sha256_check}) does not match that listed in metadata document."
        )


def download_raw_dataset(
    metadata: Dict, dl_dirname: Path, gdrive: bool = False
) -> None:
    dl_dirname.mkdir(parents=True, exist_ok=True)

    # Download multiple files
    if isinstance(metadata["filename"], list):
        for sha256, filename, url in zip(
            metadata["sha256"], metadata["filename"], metadata["url"]
        ):
            check_and_download_url(dl_dirname, sha256, filename, url, gdrive)

    # Download single file
    else:
        check_and_download_url(
            dl_dirname,
            metadata["sha256"],
            metadata["filename"],
            metadata["url"],
            gdrive,
        )
