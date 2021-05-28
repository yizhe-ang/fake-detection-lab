"""Adaptation of the LOTS algorithm for patch-based features

- Rozsa, A., Zhong, Z., & Boult, T. (2020). Adversarial Attack on Deep Learning-Based Splice Localization. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2757-2765.
- https://arxiv.org/abs/2004.08443
"""
from typing import Any, Dict

import numpy as np
import torch
from src.structures import PatchedImage
from tqdm import tqdm


class PatchLOTS:
    def __init__(
        self,
        step_size=5,
        n_iter=50,
        feat_batch_size=32,
        # pred_batch_size=1024,
        method="mean",  # One of {'mean', 'sample'}
    ) -> None:
        """
        Parameters
        ----------
        step_size : int, optional
            Step size of gradient update, by default 5
        n_iter : int, optional
            Number of gradient updates, by default 50
        feat_batch_size : int, optional
            , by default 32
        method : str, optional
            One of {'mean', 'sample'}, by default "mean"
        """
        self.step_size = step_size
        self.n_iter = n_iter
        self.feat_batch_size = feat_batch_size
        self.method = method

    def __call__(
        self,
        model,
        data: Dict[str, Any],
    ) -> torch.ByteTensor:
        """
        Parameters
        ----------
        model : [type]
        data : Dict[str, Any]
            From dataloader

        Returns
        -------
        torch.ByteTensor
            [C, H, W], the adversarially perturbed image
        """
        # Make a copy cos will be modifying it
        img = data["img"].detach().clone()
        gt_map = data["map"]

        # Perform prediction on clean image
        # print("Performing prediction on clean image...")
        # clean_preds = model.predict(img)

        print("Performing adversarial attack...")

        img = model.init_img(img)
        n_patches = img.max_h_idx * img.max_w_idx

        # Get patch features, [N, D]
        patch_feats = model.get_patch_feats(img, batch_size=self.feat_batch_size)

        # Identify all authentic patches
        # Find patches with no overlap with spliced regions
        auth_feats, is_auth = self._get_auth_feats(
            img, gt_map, patch_feats, self.feat_batch_size
        )
        if len(auth_feats) == 0:
            return img.data.detach().clone().round().byte().cpu()

        # Determine target features
        if self.method == "mean":
            # Get mean feature representation, t, of all authentic patches
            # [N, D]
            target_feats = auth_feats.mean(dim=0, keepdim=True).expand(n_patches, -1)
        elif self.method == "sample":
            # Instead of fixing a target feature
            # Model feature distribution of authentic regions
            # Make features of spliced region close to that distribution
            # Simply perform sampling?

            target_feats = patch_feats
            n_auths = len(auth_feats)
            n_not_auths = len(target_feats) - n_auths

            # Sample auth features
            sample_idx = torch.randint(n_auths, size=(n_not_auths,))

            # Replace not-auth regions with samples from auth regions
            target_feats[~is_auth] = auth_feats[sample_idx]

        # Make all the patches close to target features
        # Compute perturbation for each patch

        # Cache the best perturbed image thus far
        best_img = img.data.detach().clone()
        best_loss = float("inf")

        for _ in tqdm(range(self.n_iter)):
            # FIXME Normalize image instead? Then step size will be smaller
            img.data.requires_grad = True

            total_loss = 0

            # Have to split patches into batches (to fit in GPU memory)
            for i, patches in enumerate(img.patches_gen(self.feat_batch_size)):
                patch_feats = model.net(patches)
                # If missing batch dimension
                if len(patch_feats.shape) == 1:
                    patch_feats = patch_feats.view(1, -1)

                # Compute distance from target feature
                curr_target_feats = target_feats[
                    i * self.feat_batch_size : (i + 1) * self.feat_batch_size
                ]
                adv_loss_per_patch = ((curr_target_feats - patch_feats) ** 2).sum(
                    -1
                ) / 2
                adv_loss = adv_loss_per_patch.sum()

                # FIXME How to combine gradients from overlapping patches?
                # Just accumulate? Have to normalize?
                adv_loss.backward()

                total_loss += adv_loss.detach()

            # Perform update
            img_grad = img.data.grad.detach()
            with torch.no_grad():
                grad_norm = torch.linalg.norm(img.data.flatten(), ord=float("inf"))
                img.data = img.data - self.step_size * (img_grad / grad_norm)

                # Clip pixels
                img.data = img.data.clamp(0, 255)

            # Reset gradients
            img.data.grad = None

            # Choose the perturbed image that has features closest to target
            if total_loss < best_loss:
                # print(f"Iter {i}: Found better adversarial example")
                best_loss = total_loss
                best_img = img.data.detach().clone()

        # Round pixel values to be discrete
        adv_img = best_img.round().byte().cpu()

        # Perform prediction on adversarial image
        # print("Performing prediction on adversarial image...")
        # adv_preds = model.predict(best_img)

        # Convert into numpy image
        # adv_img = best_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # return clean_preds, adv_preds, adv_img
        return adv_img

    def _get_auth_feats(
        self,
        img: PatchedImage,
        gt_map: np.ndarray,
        patch_feats: torch.Tensor,
        batch_size=32,
    ) -> torch.Tensor:

        gt_map = torch.BoolTensor(gt_map)
        # Keep track of which patches are authentic
        is_auth = torch.zeros(img.max_h_idx * img.max_w_idx, dtype=torch.bool)

        # Find all authentic patches
        # FIXME Vectorize this
        # FIXME Put onto GPU?
        for i, patch_maps in enumerate(img.patch_maps_gen(batch_size)):
            # Check whether each patch overlaps with the spliced ground-truth
            is_auth[i * batch_size : (i + 1) * batch_size] = ~(
                (patch_maps & gt_map).flatten(1, 2).any(dim=-1)
            )

        auth_feats = patch_feats[is_auth]

        # [N_auth, D], [N,]
        return auth_feats, is_auth
