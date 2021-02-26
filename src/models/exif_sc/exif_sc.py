"""EXIF-SC overall inference model

From:
- Fighting Fake News: Image Splice Detection via Learned Self-Consistency (Huh et al., ECCV 2018)
- https://minyoungg.github.io/selfconsistency/
- https://github.com/minyoungg/selfconsistency

Network building file adapted from:
- https://github.com/Microsoft/MMdnn/blob/master/docs/tf2pytorch.md
"""
from math import floor
from typing import Any, Dict

import cv2
import numpy as np
import torch

# FIXME Something wrong with network!!
# FIXME Something wrong with image preprocessing?!!
from .networks import EXIF_Net
from .postprocess import mean_shift, normalized_cut

# TODO Check out PyTorch multiprocessing
# FIXME Careful of image shape, i.e. [C, H, W] vs [H, W, C]
# FIXME `no_grad` when running network
# FIXME Normalize image!


class EXIF_SC:
    def __init__(
        self, weight_file: str, patch_size=128, num_per_dim=30, device="cuda:0"
    ) -> None:
        """
        Parameters
        ----------
        weight_file : str
            Path to network weights file
        patch_size : int, optional
            Size of patches, by default 128
        num_per_dim : int, optional
            Number of patches to use along the largest dimension, by default 30
        device : str, optional
            , by default "cuda:0"
        """
        self.patch_size = patch_size
        self.num_per_dim = num_per_dim
        self.device = torch.device(device)

        self.net = EXIF_Net(weight_file)
        self.net.eval()
        self.net.to(device)

    def predict(
        self,
        img: torch.Tensor,
        feat_batch_size=32,  # Does not affect compute time much?
        pred_batch_size=1024,  # Affects up to a certain extent
        blue_high=True,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        img : torch.Tensor
            [C, H, W], range: [0, 255]
        feat_batch_size : int, optional
            , by default 32
        pred_batch_size : int, optional
            , by default 1024
        blue_high : bool
            , by default True

        Returns
        -------
        Dict[str, Any]
            ms : np.ndarray (float32)
                Consistency map, [H, W], range [0, 1]
            ncuts : np.ndarray (float32)
                Localization map, [H, W], range [0, 1]
            score : float
                Prediction score, higher indicates existence of manipulation
        """
        _, height, width = img.shape
        assert (
            min(height, width) > self.patch_size
        ), "Image must be bigger than patch size!"

        # Initialize image and attributes
        self._init_img(img)

        # Precompute features for each patch
        patch_features = self._get_patch_features(batch_size=feat_batch_size)

        # Predict consistency maps
        pred_maps = self._predict_consistency_maps(
            patch_features, batch_size=pred_batch_size
        )

        # Produce a single response map
        ms = mean_shift(
            pred_maps.reshape((-1, pred_maps.shape[0] * pred_maps.shape[1])), pred_maps
        )

        # As a heuristic, the anomalous areas are smaller than the normal areas
        if np.mean(ms > 0.5) > 0.5:
            # majority of the image is above .5
            if blue_high:
                # Reverse heat map
                ms = 1 - ms

        # Run clustering to get localization map
        ncuts = normalized_cut(pred_maps)
        if np.mean(ncuts > 0.5) > 0.5:
            # majority of the image is white
            # flip so spliced is white
            ncuts = 1 - ncuts
        out_ncuts = cv2.resize(
            ncuts.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        out_ms = cv2.resize(ms, (width, height), interpolation=cv2.INTER_LINEAR)

        # FIXME Use out_ms or out_ncuts for prediction score?
        return {"ms": out_ms, "ncuts": out_ncuts, "score": out_ms.mean()}

    def _predict_consistency_maps(self, patch_features, batch_size=64):
        # For each patch, how many overlapping patches?
        spread = max(1, self.patch_size // self.stride)

        # Aggregate prediction maps; for each patch, compared to each other patch
        responses = torch.zeros(
            (
                self.max_h_idx + spread - 1,
                self.max_w_idx + spread - 1,
                self.max_h_idx + spread - 1,
                self.max_w_idx + spread - 1,
            )
        )
        # Number of predictions for each patch
        vote_counts = (
            torch.zeros(
                (
                    self.max_h_idx + spread - 1,
                    self.max_w_idx + spread - 1,
                    self.max_h_idx + spread - 1,
                    self.max_w_idx + spread - 1,
                )
            )
            + 1e-4
        )

        # Perform prediction
        for idxs in self._pred_idxs_gen(batch_size=batch_size):
            # a to be compared to b
            patch_a_idxs = idxs[:, :2]  # [B, 2]
            patch_b_idxs = idxs[:, 2:]  # [B, 2]

            # Convert 2D index into its 1D version
            a_idxs = torch.from_numpy(
                np.ravel_multi_index(patch_a_idxs.T, [self.max_h_idx, self.max_w_idx])
            )  # [B]
            b_idxs = torch.from_numpy(
                np.ravel_multi_index(patch_b_idxs.T, [self.max_h_idx, self.max_w_idx])
            )

            # Grab corresponding features
            a_feats = patch_features[a_idxs]  # [B, 4096]
            b_feats = patch_features[b_idxs]

            feats = torch.cat([a_feats, b_feats], dim=-1)  # [B, 8192]

            # Get predictions
            with torch.no_grad():
                exif_logits = self.net.exif_fc(feats)  # [B, 83]
                # FIXME Sigmoid or nay?
                # exif_preds = torch.sigmoid(exif_logits)
                exif_preds = exif_logits

                logits = self.net.classifier_fc(exif_preds)
                preds = torch.sigmoid(logits)  # [B, 1]

            preds = preds.cpu()

            # FIXME Is it possible to vectorize this?
            # Accumulate predictions for overlapping patches
            for i in range(len(preds)):
                responses[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    idxs[i][2] : (idxs[i][2] + spread),
                    idxs[i][3] : (idxs[i][3] + spread),
                ] += preds[i]
                vote_counts[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    idxs[i][2] : (idxs[i][2] + spread),
                    idxs[i][3] : (idxs[i][3] + spread),
                ] += 1

        # Normalize predictions
        return responses / vote_counts

    def _init_img(self, img: torch.Tensor) -> None:
        """Initialize image attributes, i.e. stride, number of patches

        Parameters
        ----------
        img : torch.Tensor
        """
        # Preprocess and set img attribute
        self.img = self.preprocess_img(img.to(self.device))
        _, height, width = img.shape

        # Compute patch stride on image
        self.stride = (max(height, width) - self.patch_size) // self.num_per_dim

        # Compute total number of patches along height and width dimension
        self.max_h_idx = 1 + floor((height - self.patch_size) / self.stride)
        self.max_w_idx = 1 + floor((width - self.patch_size) / self.stride)

    def _get_patch(self, h_idx: int, w_idx: int) -> torch.Tensor:
        """Get a patch from the image

        Parameters
        ----------
        h_idx : int
        w_idx : int

        Returns
        -------
        torch.Tensor
            [3, patch_size, patch_size]
        """
        h_coord = h_idx * self.stride
        w_coord = w_idx * self.stride

        return self.img[
            :, h_coord : h_coord + self.patch_size, w_coord : w_coord + self.patch_size
        ]

    def _get_patches(self, idxs: torch.Tensor) -> torch.Tensor:
        """Get patches from image given its indices

        Parameters
        ----------
        idxs : torch.Tensor
            [n_patches, 2], [n_patches, (h_idx, w_idx)]

        Returns
        -------
        torch.Tensor
            [n_patches, 3, patch_size, patch_size]
        """
        n_patches = idxs.shape[0]
        patches = torch.zeros(
            n_patches, 3, self.patch_size, self.patch_size, device=self.device
        )

        # FIXME Any way to vectorize this?
        # https://discuss.pytorch.org/t/advanced-fancy-indexing-across-batches/103445
        for i, idx in enumerate(idxs):
            h_idx, w_idx = idx

            patches[i] = self._get_patch(h_idx, w_idx)

        return patches

    def _patches_gen(self, batch_size=32) -> torch.Tensor:
        """Generator for all patches in an image, in raster scan order

        Parameters
        ----------
        batch_size : int, optional
            Number of patches in each iteration, by default 32

        Returns
        -------
        torch.Tensor
            [batch_size, 3, patch_size, patch_size]
        """
        count = 0

        # Initialize indices / coords of all patches, [n_patches, 2]
        h_idxs = torch.arange(self.max_h_idx)
        w_idxs = torch.arange(self.max_w_idx)
        idxs = torch.stack(torch.meshgrid([h_idxs, w_idxs])).view(2, -1).T

        n_patches = len(idxs)

        while True:
            # Break when run out of patches
            if count * batch_size >= n_patches:
                break

            # Yield a batch of patches
            patches = self._get_patches(
                idxs[count * batch_size : (count + 1) * batch_size]
            )
            yield patches
            count += 1

    def _pred_idxs_gen(self, batch_size=32) -> torch.Tensor:
        """Generator for all prediction map indices

        Parameters
        ----------
        batch_size : int, optional
            Number of indices in each iteration, by default 32

        Returns
        -------
        torch.Tensor
            [batch_size, 4]
        """
        # h_idxs = torch.arange(self.max_h_idx)
        # w_idxs = torch.arange(self.max_w_idx)

        # # All possible pairs of patches, [n_idxs, 4]
        # # n_idxs: max_h_ind ^ 2 * max_w_ind ^ 2
        # idxs = (
        #     torch.stack(torch.meshgrid([h_idxs, w_idxs, h_idxs, w_idxs])).view(4, -1).T
        # )

        # # All possible pairs of patches, [n_idxs, 4]
        # # n_idxs: max_h_ind ^ 2 * max_w_ind ^ 2
        idxs = (
            np.mgrid[
                0 : self.max_h_idx,
                0 : self.max_w_idx,
                0 : self.max_h_idx,
                0 : self.max_w_idx,
            ]
            .reshape((4, -1))
            .T
        )

        count = 0
        while True:
            if count * batch_size >= len(idxs):
                break

            yield idxs[count * batch_size : (count + 1) * batch_size]
            count += 1

    def _get_patch_features(self, n_features=4096, batch_size=32) -> torch.Tensor:
        """Get features for every patch in the image

        Parameters
        ----------
        n_features : int, optional
            Feature vector size for each patch, by default 4096
        batch_size : int, optional
            Batch size to be fed into the network, by default 32

        Returns
        -------
        torch.Tensor
            [n_patches, n_features]
        """
        # Compute feature vector for each image patch
        patch_features = []

        # Generator for patches; raster scan order
        with torch.no_grad():
            for patches in self._patches_gen(batch_size):
                feat = self.net(patches)
                # If missing batch dimension
                if len(feat.shape) == 1:
                    feat = feat.view(1, -1)
                patch_features.append(feat)

        # [n_patches, n_features]
        patch_features = torch.cat(patch_features, dim=0)

        # Try preallocate tensor instead
        # patch_features = torch.zeros(self.max_h_idx * self.max_w_idx, n_features)

        # with torch.no_grad():
        #     for i, patches in enumerate(self._patches_gen(batch_size)):
        #         patch_features[i * batch_size : (i + 1) * batch_size] = self.net(
        #             patches
        #         )

        # patch_features = (patch_features.T).view(
        #     n_features, self.max_h_idx, self.max_w_idx
        # )

        return patch_features

    @staticmethod
    def preprocess_img(img: torch.Tensor) -> torch.Tensor:
        """ Normalizes images into the range [-1.0, 1.0] """
        if img.max() <= 1:
            # PNG format
            img = (2.0 * img) - 1.0
        else:
            # JPEG format
            img = 2.0 * (img / 255.0) - 1.0
        return img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_path",
        help="path to the weights file",
        default="artifacts/exif_sc.npy",
    )
    parser.add_argument(
        "--img_path",
        help="path to the input image file",
        default="data/demo.png",
    )
    args = parser.parse_args()

    model = EXIF_SC(args.weights_path)

    img = cv2.imread(args.img_path)[:, :, [2, 1, 0]]  # [H, W, C]
    img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
