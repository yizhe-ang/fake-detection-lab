from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, dataset: Dataset) -> None:
        self.model = model
        self.dataset = dataset

        self.metrics = {}

    def evaluate(self, save=False, resize: Tuple[int, int] = None) -> Dict[str, Any]:
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
        """
        y_true = []
        label_map = []

        y_score = []
        score_map = []
        ncut = []

        for i in tqdm(range(len(self.dataset))):
            data = self.dataset[i]
            # Perform prediction
            pred = self.model.predict(data["img"])

            # If image sizes different, resize to a consistent shape
            if resize:
                data["map"] = cv2.resize(
                    data["map"], resize, interpolation=cv2.INTER_LINEAR
                )
                pred["ms"] = cv2.resize(
                    pred["ms"], resize, interpolation=cv2.INTER_LINEAR
                )

            # Store ground-truths
            y_true.append(data["label"])
            label_map.append(data["map"])

            # Store predictions
            y_score.append(pred["score"])
            score_map.append(pred["ms"])
            # ncut.append(pred["ncuts"])

        y_true = np.array(y_true)
        label_map = np.stack(label_map, axis=0)

        y_score = np.array(y_score)
        score_map = np.stack(score_map, axis=0)
        # ncut = np.stack(ncut, axis=0)

        # Save predictions
        if save:
            save_path = Path("artifacts/predictions")
            np.save(save_path / "scores.npy", y_score)
            np.save(save_path / "score_maps.npy", score_map)
            # np.save(save_path / "rt_ncuts.npy", ncut)

        # Compute localization metrics
        self._compute_localization_metrics(label_map, score_map)

        # Compute detection metrics
        self._compute_detection_metrics(y_true, y_score)

        return self.metrics

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

    def _compute_localization_metrics(self, label_map, score_map) -> None:
        # Check whether NaN values
        if np.isnan(score_map).any():
            print("WARNING: NaN values in localization prediction scores!")
            score_map[np.isnan(score_map)] = 0

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

        self.metrics["IoU-spliced"] = iou_spliced
        self.metrics["IoU-non-spliced"] = iou_non_spliced
        # Compute mean IoU
        self.metrics["IoU"] = (iou_spliced + iou_non_spliced) / 2

        # FIXME Report per-class scores

    def _compute_detection_metrics(self, y_true, y_score) -> None:
        # Check whether NaN values
        if np.isnan(y_score).any():
            print("WARNING: NaN values in detection prediction scores!")
            y_score[np.isnan(y_score)] = 0

        self.metrics["AP"] = average_precision_score(y_true, y_score)

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
