from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from src.attacks import PatchLOTS
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from .metrics import AUC_Metric, F1_Metric, MCC_Metric, mAP_Metric


class Evaluator:
    def __init__(
        self,
        model,
        dataset: Dataset,
        adv_step_size: int,
        adv_n_iter: int,
        vis_dir: str = None,
        vis_every=1,
        logger=None,
        method="mean",
    ) -> None:
        # Freeze all network weights
        for parameter in model.net.parameters():
            parameter.requires_grad = False
        model.net.eval()
        self.model = model

        self.dataset = dataset
        self.adv_step_size = adv_step_size
        self.adv_n_iter = adv_n_iter
        self.method = method

        self.vis_dir = vis_dir
        self.vis_every = vis_every
        self.logger = logger

        self.results = {"clean": {}, "adv": {}}
        self.attacker = PatchLOTS()

    def __call__(self, resize: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Parameters
        ----------
        save : bool, optional
            Whether to save prediction arrays, by default False
        resize : Tuple[int, int], optional
            [H, W], whether to resize images / maps to a consistent shape

        Returns
        -------
        Dict[str, Any]
            AP : float
                Average precision score, for detection
            IoU : float
                Class-balanced IoU, for localization
            f1_score : float
                for localization
            mcc : float
                Matthews Correlation Coefficient, for localization
            mAP : float
                Mean Average Precision, for localization
            auc : float
                Area under the Receiving Operating Characteristic Curve, for localization
        """
        # Initialize per-image metrics
        metrics = defaultdict(dict)
        metric_classes = {
            "f1_score": F1_Metric,
            "mcc": MCC_Metric,
            "mAP": mAP_Metric,
            "auc": AUC_Metric,
        }
        for type in ["clean", "adv"]:
            for name, cls in metric_classes.items():
                metrics[type][name] = cls()

        # Cache all predictions
        all_preds = {"clean": defaultdict(list), "adv": defaultdict(list)}

        # Loop through dataset
        for i in tqdm(range(len(self.dataset))):
            # Store per-image predictions
            img_pred = {}

            data = self.dataset[i]
            clean_img = data["img"]

            # Perform prediction on clean image
            img_pred["clean"] = self.model.predict(clean_img)

            # Generate adversarial image
            adv_img = self.attacker(
                self.model,
                data,
                self.adv_step_size,
                self.adv_n_iter,
                method=self.method,
            )

            # Perform prediction on adversarial image
            img_pred["adv"] = self.model.predict(adv_img)

            # Account for NaN values
            for pred in img_pred.values():
                if np.isnan(pred["ms"]).any():
                    print("WARNING: NaN values in localization prediction scores!")
                    pred["ms"][np.isnan(pred["ms"])] = 0

                if np.isnan(pred["score"]):
                    print("WARNING: NaN values in detection prediction scores!")
                    pred["score"] = 0

            # Perform per-image evaluations
            for type, ms in metrics.items():
                for _, m in ms.items():
                    m.update(data["map"], img_pred[type]["ms"])

            # Visualize some examples
            if self.vis_dir and i % self.vis_every == 0:
                self._vis_preds(i, data, img_pred, clean_img, adv_img)

            # If image sizes different, resize to a consistent shape
            if resize:
                data["map"] = cv2.resize(
                    data["map"], resize[::-1], interpolation=cv2.INTER_LINEAR
                )
                img_pred["clean"]["ms"] = cv2.resize(
                    img_pred["clean"]["ms"],
                    resize[::-1],
                    interpolation=cv2.INTER_LINEAR,
                )
                img_pred["adv"]["ms"] = cv2.resize(
                    img_pred["adv"]["ms"], resize[::-1], interpolation=cv2.INTER_LINEAR
                )

            # Cache predictions
            for type, preds in all_preds.items():
                # Store ground-truths
                preds["y_true"].append(data["label"])
                preds["label_map"].append(data["map"])

                # Store predictions
                preds["y_score"].append(img_pred[type]["score"])
                preds["score_map"].append(img_pred[type]["ms"])

        # Compute per-image evaluation metrics
        for type, ms in metrics.items():
            for metric_name, m in ms.items():
                self.results[type][metric_name] = m.compute()

        # Consolidate cached predictions
        for type, preds in all_preds.items():
            preds["y_true"] = np.array(preds["y_true"])
            preds["label_map"] = np.stack(preds["label_map"], axis=0)

            preds["y_score"] = np.array(preds["y_score"])
            preds["score_map"] = np.stack(preds["score_map"], axis=0)

        # Save predictions
        # if save:
        #     save_path = Path("artifacts/predictions")
        #     np.save(save_path / "scores.npy", y_score)
        #     np.save(save_path / "score_maps.npy", score_map)
        # np.save(save_path / "rt_ncuts.npy", ncut)

        # Compute rest of the metrics on cached predictions
        for type, r in self.results.items():
            # Compute per-class IoU
            iou_spliced, iou_non_spliced, iou = self._compute_class_iou(
                all_preds[type]["label_map"], all_preds[type]["score_map"]
            )
            r["iou_spliced"] = iou_spliced
            r["iou_non_spliced"] = iou_non_spliced
            r["iou"] = iou

            # Compute detection metrics
            r["AP"] = average_precision_score(
                all_preds[type]["y_true"], all_preds[type]["y_score"]
            )

        return self.results

        # @staticmethod
        # def compute_optimal_iou(y_true, y_pred, batch_size=256):
        #     # Check whether NaN values
        #     if np.isnan(y_pred).any():
        #         print("WARNING: NaN values in localization prediction scores!")
        #         y_pred[np.isnan(y_pred)] = 0

        #     # Store all possible iou scores
        #     thresholds = y_pred.flatten()
        #     scores = np.zeros_like(thresholds)

        #     for i in range(0, len(scores), batch_size):
        #         threshs = thresholds[i : i + batch_size]  # [B]

        #         y_preds = y_pred.copy()
        #         # [H, W, B]
        #         y_preds = np.repeat(y_preds[..., None], batch_size, axis=-1)

        #         # Perform thresholding
        #         y_preds[y_preds < threshs] = 0
        #         y_preds[y_preds >= threshs] = 1

        #         # Compute scores
        #         return iou(
        #             torch.from_numpy(y_preds.transpose(2, 0, 1)),
        #             torch.from_numpy(np.repeat(y_true[None, ...], batch_size, axis=0))
        #         )

        # Compute iou score for each threshold
        # for i, thresh in tqdm(enumerate(thresholds)):
        #     y_pred_thresh = np.zeros(y_pred.shape, dtype=np.uint8)
        #     y_pred_thresh[y_pred >= thresh] = 1

        #     scores[i] = jaccard_score(y_true.flatten(), y_pred_thresh.flatten())

        return scores

    def _compute_class_iou(self, label_map, score_map) -> None:
        # FIXME Consider inverted score maps?

        # Find optimal threshold, and the corresponding score for each image

        # Compute for spliced regions
        _, iou_spliced = self.find_optimal_threshold(score_map, label_map)
        iou_spliced = iou_spliced.mean().item()

        # Compute for non-spliced regions
        invert_label_map = 1 - label_map
        invert_score_map = 1 - score_map

        _, iou_non_spliced = self.find_optimal_threshold(
            invert_score_map, invert_label_map
        )
        iou_non_spliced = iou_non_spliced.mean().item()

        # Compute mean IoU
        iou = (iou_spliced + iou_non_spliced) / 2

        # FIXME Report per-class scores

        return iou_spliced, iou_non_spliced, iou

    @staticmethod
    def find_optimal_threshold(
        pred_mask: np.ndarray, groundtruth_masks: np.ndarray
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """https://codereview.stackexchange.com/questions/229341/pytorch-vectorized-implementation-for-thresholding-and-computing-jaccard-index

        Parameters
        ----------
        pred_mask : np.ndarray (float32)
            [B, H, W], range [0, 1], probability prediction map
        groundtruth_masks : np.ndarray (uint8)
            [B, H, W], values one of {0, 1}, binary label map

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor]
            [B], optimal thresholds for each image
            [B], corresponding jaccard scores for each image
        """
        n_patch = groundtruth_masks.shape[0]

        groundtruth_masks_tensor = torch.from_numpy(groundtruth_masks)
        pred_mask_tensor = torch.from_numpy(pred_mask)

        # if USE_CUDA:
        #     groundtruth_masks_tensor = groundtruth_masks_tensor.cuda()
        #     pred_mask_tensor = pred_mask_tensor.cuda()

        vector_pred = pred_mask_tensor.view(n_patch, -1)
        vector_gt = groundtruth_masks_tensor.view(n_patch, -1)
        vector_pred, sort_pred_idx = torch.sort(vector_pred, descending=True)
        vector_gt = vector_gt[torch.arange(vector_gt.shape[0])[:, None], sort_pred_idx]
        gt_cumsum = torch.cumsum(vector_gt, dim=1)
        gt_total = gt_cumsum[:, -1].reshape(n_patch, 1)
        predicted = torch.arange(start=1, end=vector_pred.shape[1] + 1)
        # if USE_CUDA:
        #     predicted = predicted.cuda()
        gt_cumsum = gt_cumsum.type(torch.float)
        gt_total = gt_total.type(torch.float)
        predicted = predicted.type(torch.float)
        jaccard_idx = gt_cumsum / (gt_total + predicted - gt_cumsum)
        max_jaccard_idx, max_indices = torch.max(jaccard_idx, dim=1)
        max_indices = max_indices.reshape(-1, 1)
        best_threshold = vector_pred[
            torch.arange(vector_pred.shape[0])[:, None], max_indices
        ]
        best_threshold = best_threshold.reshape(-1)

        return best_threshold, max_jaccard_idx

    def _vis_preds(self, i, data, img_pred, clean_img, adv_img):
        plt.subplots(figsize=(32, 8))
        plt.subplot(2, 4, 1)
        plt.title("Input Image")
        plt.imshow(clean_img.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(2, 4, 2)
        plt.title("Adv Image")
        plt.imshow(adv_img.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(2, 4, 3)
        plt.title("Cluster w/ MeanShift")
        plt.axis("off")
        plt.imshow(img_pred["clean"]["ms"], cmap="jet", vmin=0.0, vmax=1.0)

        plt.subplot(2, 4, 4)
        plt.title("Adv Cluster w/ MeanShift")
        plt.axis("off")
        plt.imshow(img_pred["adv"]["ms"], cmap="jet", vmin=0.0, vmax=1.0)

        plt.subplot(2, 4, 5)
        plt.title("Ground-truth Segment")
        plt.axis("off")
        plt.imshow(data["map"], vmin=0.0, vmax=1.0, cmap="gray")

        plt.subplot(2, 4, 6)
        plt.title("Ground-truth Segment")
        plt.axis("off")
        plt.imshow(data["map"], vmin=0.0, vmax=1.0, cmap="gray")

        plt.subplot(2, 4, 7)
        plt.title("Segment with NCuts")
        plt.axis("off")
        plt.imshow(img_pred["clean"]["ncuts"], vmin=0.0, vmax=1.0, cmap="gray")

        plt.subplot(2, 4, 8)
        plt.title("Adv Segment with NCuts")
        plt.axis("off")
        plt.imshow(img_pred["adv"]["ncuts"], vmin=0.0, vmax=1.0, cmap="gray")

        plt.tight_layout()
        plt.show()

        vis_dir = Path(self.vis_dir)
        plt.savefig(vis_dir / f"{self.dataset.__class__.__name__}_{i}.png")

        if self.logger:
            self.logger.log({"adv_example": self.logger.Image(plt)})
