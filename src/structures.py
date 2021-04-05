from math import floor

import numpy as np
import torch


class PatchedImage:
    def __init__(
        self,
        data: torch.Tensor,
        patch_size=128,
        num_per_dim=30,
    ) -> None:
        """Representation of an image that is sliced into patches"""
        self.data = data.float()

        # Initialize image attributes
        self.patch_size = patch_size
        self.num_per_dim = num_per_dim

        self.shape = data.shape
        _, height, width = data.shape

        # Compute patch stride on image
        self.stride = (max(height, width) - self.patch_size) // self.num_per_dim

        # Compute total number of patches along height and width dimension
        self.max_h_idx = 1 + floor((height - self.patch_size) / self.stride)
        self.max_w_idx = 1 + floor((width - self.patch_size) / self.stride)

    def get_patch(self, h_idx: int, w_idx: int) -> torch.Tensor:
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

        return self.data[
            :, h_coord : h_coord + self.patch_size, w_coord : w_coord + self.patch_size
        ]

    def get_patch_map(self, h_idx: int, w_idx: int) -> torch.ByteTensor:
        """
        Parameters
        ----------
        h_idx : int
        w_idx : int

        Returns
        -------
        torch.ByteTensor
            [H, W], values of {0, 1}
        """
        h_coord = h_idx * self.stride
        w_coord = w_idx * self.stride

        _, height, width = self.shape

        binary_map = torch.zeros(height, width, dtype=torch.bool)
        binary_map[
            h_coord : h_coord + self.patch_size, w_coord : w_coord + self.patch_size
        ] = True

        return binary_map

    def get_patches(self, idxs: torch.Tensor) -> torch.Tensor:
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
            n_patches, 3, self.patch_size, self.patch_size, device=self.data.device
        )

        # FIXME Any way to vectorize this?
        # https://discuss.pytorch.org/t/advanced-fancy-indexing-across-batches/103445
        for i, idx in enumerate(idxs):
            h_idx, w_idx = idx

            patches[i] = self.get_patch(h_idx, w_idx)

        return patches

    def get_patch_maps(self, idxs: torch.Tensor) -> torch.Tensor:
        n_patches = idxs.shape[0]
        _, height, width = self.shape

        maps = torch.zeros(n_patches, height, width, dtype=torch.bool)

        for i, idx in enumerate(idxs):
            h_idx, w_idx = idx

            maps[i] = self.get_patch_map(h_idx, w_idx)

        return maps

    def patches_gen(self, batch_size=32) -> torch.Tensor:
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
            patches = self.get_patches(
                idxs[count * batch_size : (count + 1) * batch_size]
            )
            yield patches
            count += 1

    def patch_maps_gen(self, batch_size=32) -> torch.Tensor:
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
            patches = self.get_patch_maps(
                idxs[count * batch_size : (count + 1) * batch_size]
            )
            yield patches
            count += 1

    def pred_idxs_gen(self, batch_size=32) -> torch.Tensor:
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
