from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule
from torchmetrics import Accuracy


class EXIF_Trainer1(LightningModule):
    def __init__(
        self, net: nn.Module, datamodule: LightningDataModule, config: Dict[str, Any]
    ) -> None:
        super().__init__()

        self.net = net
        self.dm = datamodule
        self.config = config

        # Initialize metrics
        self.exif_attrs = self.dm.exif_attrs
        self.metrics = nn.ModuleList([Accuracy() for _ in range(len(self.exif_attrs))])

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.config["learning_rate_1"])

        return optimizer

    def training_step(self, batch, batch_idx):
        # [2, B, C, H, W], [B, n_exif_attr]
        imgs, labels = batch

        _, B, C, H, W = imgs.shape
        imgs = imgs.view(-1, C, H, W)  # [2*B, C, H, W]

        feats = self.net(imgs)  # [B*2, 4096]
        feats = torch.cat([feats[:B], feats[B:]], dim=-1)  # [B, 8192]

        logits = self.net.exif_fc(feats)  # [B, n_exif_attr]

        labels_float = labels.float()
        loss = F.binary_cross_entropy_with_logits(logits, labels_float)

        # Log metrics
        self.log("train/loss_1", loss, prog_bar=True)

        # Compute accuracy for each attr
        with torch.no_grad():
            probs = torch.sigmoid(logits)

            metrics_dict = {}
            for i, (attr, m) in enumerate(zip(self.exif_attrs, self.metrics)):
                m(probs[:, i], labels[:, i])
                metrics_dict[f"val/{attr}_acc"] = m

            self.log_dict(metrics_dict, on_step=False, on_epoch=True)

        return loss


class EXIF_Trainer2(LightningModule):
    def __init__(
        self, net: nn.Module, datamodule: LightningDataModule, config: Dict[str, Any]
    ) -> None:
        super().__init__()

        # Freeze entire network except for final classification MLP
        for name, params in net.named_parameters():
            if "classifier_fc" not in name:
                params.requires_grad = False
        self.net = net

        self.dm = datamodule
        self.config = config

        # Initialize metrics
        self.acc = Accuracy()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.config["learning_rate_2"])

        return optimizer

    def training_step(self, batch, batch_idx):
        # [2, B, C, H, W], [B,]
        imgs, labels = batch

        _, B, C, H, W = imgs.shape
        imgs = imgs.view(-1, C, H, W)  # [2*B, C, H, W]

        feats = self.net(imgs)  # [B*2, 4096]
        feats = torch.cat([feats[:B], feats[B:]], dim=-1)  # [B, 8192]

        logits = self.net.exif_fc(feats)  # [B, n_exif_attr]
        binary_logit = self.net.classifier_fc(logits).view(-1)  # [B,]

        labels_float = labels.float()
        loss = F.binary_cross_entropy_with_logits(binary_logit, labels_float)

        # Log metrics
        self.log("train/loss_2", loss, prog_bar=True)

        # Compute accuracy for image pair prediction
        with torch.no_grad():
            prob = torch.sigmoid(binary_logit)

            self.acc(prob, labels)
            self.log("val/img_pred_acc", self.acc, on_step=False, on_epoch=True)

        return loss
