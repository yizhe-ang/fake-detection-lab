from io import BytesIO
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image


class JPEG_Compressor:
    def __init__(self, quality: int = 30) -> None:
        """
        Parameters
        ----------
        quality : int, optional
            Quality of compressed image, from [0, 100], by default 30
        """
        self.quality = quality

    def __call__(self, model, data: Dict[str, Any]) -> torch.ByteTensor:
        """
        Parameters
        ----------
        model : [type]
        data : Dict[str, Any]
            From dataloader

        Returns
        -------
        torch.ByteTensor
            [C, H, W], the compressed tensor
        """
        clean_img = data["img"]
        # Convert to PIL image
        np_img = clean_img.permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(np_img)

        # Compress to JPEG
        with BytesIO() as f:
            pil_img.save(f, format="JPEG", optimize=True, quality=self.quality)
            f.seek(0)
            jpg_img = Image.open(f)
            jpg_img.load()

        # Convert back to torch tensor
        jpg_img_np = np.array(jpg_img)
        jpg_img_t = torch.tensor(jpg_img_np).permute(2, 0, 1)

        return jpg_img_t
