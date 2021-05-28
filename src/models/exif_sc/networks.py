"""EXIF-SC network model

From:
- Fighting Fake News: Image Splice Detection via Learned Self-Consistency (Huh et al., ECCV 2018)
- https://minyoungg.github.io/selfconsistency/
- https://github.com/minyoungg/selfconsistency

Network building file adapted from:
- https://github.com/Microsoft/MMdnn/blob/master/docs/tf2pytorch.md
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_weights_dict = dict()


def load_weights(weight_file, weight_ext):
    if weight_ext != "npy":
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding="bytes").item()

    print("Loaded numpy weights.")

    return weights_dict


class EXIF_Net(nn.Module):
    def __init__(self, weight_file: str = None, n_attrs: int = 83):
        super().__init__()

        # Get format of weights
        weight_ext = None if weight_file is None else weight_file.split(".")[-1]

        # Load numpy weights
        global _weights_dict
        _weights_dict = load_weights(weight_file, weight_ext)

        # EXIF prediction network
        self.exif_fc = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_attrs),
        )
        self.__set_fc_params(self.exif_fc[0], "predict/fc/fc_1")
        self.__set_fc_params(self.exif_fc[2], "predict/fc/fc_2")
        self.__set_fc_params(self.exif_fc[4], "predict/fc/fc_3")
        self.__set_fc_params(self.exif_fc[6], "predict/fc_out")

        # Classification network
        self.classifier_fc = nn.Sequential(
            nn.Linear(n_attrs, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.__set_fc_params(self.classifier_fc[0], "classify/fc/fc_1")
        self.__set_fc_params(self.classifier_fc[2], "classify/fc_out")

        # ResNet backbone
        self._init_backbone()

        # HACK Load torch ckpt weights
        if weight_ext == "ckpt":
            ckpt = torch.load(weight_file)

            # Rename parameter keys
            new_state_dict = {}
            for name, params in ckpt["state_dict"].items():
                new_state_dict[name[4:]] = params

            self.load_state_dict(new_state_dict)

            print("Loaded torch checkpoint.")

    @staticmethod
    def preprocess_img(img: torch.Tensor) -> torch.Tensor:
        """Normalizes images into the range [-1.0, 1.0]"""
        # if img.max() <= 1:
        #     # PNG format
        #     img = (2.0 * img) - 1.0
        # else:

        # JPEG format
        img = 2.0 * (img / 255.0) - 1.0

        return img

    def _init_backbone(self):
        self.resnet_v2_50_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/conv1",
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block1_unit_1_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block1/unit_1/bottleneck_v2/preact",
                num_features=64,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block1_unit_1_bottleneck_v2_shortcut_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block1/unit_1/bottleneck_v2/shortcut",
            in_channels=64,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block1_unit_1_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block1/unit_1/bottleneck_v2/conv1",
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block1_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm",
                num_features=64,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block1_unit_1_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block1/unit_1/bottleneck_v2/conv2",
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block1_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block1/unit_1/bottleneck_v2/conv2/BatchNorm",
                num_features=64,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block1_unit_1_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block1/unit_1/bottleneck_v2/conv3",
            in_channels=64,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block1_unit_2_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block1/unit_2/bottleneck_v2/preact",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block1_unit_2_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block1/unit_2/bottleneck_v2/conv1",
            in_channels=256,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block1_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block1/unit_2/bottleneck_v2/conv1/BatchNorm",
                num_features=64,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block1_unit_2_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block1/unit_2/bottleneck_v2/conv2",
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block1_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block1/unit_2/bottleneck_v2/conv2/BatchNorm",
                num_features=64,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block1_unit_2_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block1/unit_2/bottleneck_v2/conv3",
            in_channels=64,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block1_unit_3_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block1/unit_3/bottleneck_v2/preact",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block1_unit_3_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block1/unit_3/bottleneck_v2/conv1",
            in_channels=256,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block1_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block1/unit_3/bottleneck_v2/conv1/BatchNorm",
                num_features=64,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block1_unit_3_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block1/unit_3/bottleneck_v2/conv2",
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2, 2),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block1_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block1/unit_3/bottleneck_v2/conv2/BatchNorm",
                num_features=64,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block1_unit_3_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block1/unit_3/bottleneck_v2/conv3",
            in_channels=64,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block2_unit_1_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block2/unit_1/bottleneck_v2/preact",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block2_unit_1_bottleneck_v2_shortcut_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_1/bottleneck_v2/shortcut",
            in_channels=256,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block2_unit_1_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_1/bottleneck_v2/conv1",
            in_channels=256,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block2_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block2/unit_1/bottleneck_v2/conv1/BatchNorm",
                num_features=128,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block2_unit_1_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_1/bottleneck_v2/conv2",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block2_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block2/unit_1/bottleneck_v2/conv2/BatchNorm",
                num_features=128,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block2_unit_1_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_1/bottleneck_v2/conv3",
            in_channels=128,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block2_unit_2_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block2/unit_2/bottleneck_v2/preact",
                num_features=512,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block2_unit_2_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_2/bottleneck_v2/conv1",
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block2_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block2/unit_2/bottleneck_v2/conv1/BatchNorm",
                num_features=128,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block2_unit_2_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_2/bottleneck_v2/conv2",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block2_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block2/unit_2/bottleneck_v2/conv2/BatchNorm",
                num_features=128,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block2_unit_2_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_2/bottleneck_v2/conv3",
            in_channels=128,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block2_unit_3_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block2/unit_3/bottleneck_v2/preact",
                num_features=512,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block2_unit_3_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_3/bottleneck_v2/conv1",
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block2_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block2/unit_3/bottleneck_v2/conv1/BatchNorm",
                num_features=128,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block2_unit_3_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_3/bottleneck_v2/conv2",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block2_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block2/unit_3/bottleneck_v2/conv2/BatchNorm",
                num_features=128,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block2_unit_3_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_3/bottleneck_v2/conv3",
            in_channels=128,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block2_unit_4_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block2/unit_4/bottleneck_v2/preact",
                num_features=512,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block2_unit_4_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_4/bottleneck_v2/conv1",
            in_channels=512,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block2_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block2/unit_4/bottleneck_v2/conv1/BatchNorm",
                num_features=128,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block2_unit_4_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_4/bottleneck_v2/conv2",
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(2, 2),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block2_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block2/unit_4/bottleneck_v2/conv2/BatchNorm",
                num_features=128,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block2_unit_4_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block2/unit_4/bottleneck_v2/conv3",
            in_channels=128,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block3_unit_1_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_1/bottleneck_v2/preact",
                num_features=512,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_1_bottleneck_v2_shortcut_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_1/bottleneck_v2/shortcut",
            in_channels=512,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block3_unit_1_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_1/bottleneck_v2/conv1",
            in_channels=512,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block3_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_1/bottleneck_v2/conv1/BatchNorm",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_1_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_1/bottleneck_v2/conv2",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block3_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_1/bottleneck_v2/conv2/BatchNorm",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_1_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_1/bottleneck_v2/conv3",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block3_unit_2_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_2/bottleneck_v2/preact",
                num_features=1024,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_2_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_2/bottleneck_v2/conv1",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block3_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_2/bottleneck_v2/conv1/BatchNorm",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_2_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_2/bottleneck_v2/conv2",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block3_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_2/bottleneck_v2/conv2/BatchNorm",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_2_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_2/bottleneck_v2/conv3",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block3_unit_3_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_3/bottleneck_v2/preact",
                num_features=1024,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_3_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_3/bottleneck_v2/conv1",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block3_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_3/bottleneck_v2/conv1/BatchNorm",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_3_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_3/bottleneck_v2/conv2",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block3_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_3/bottleneck_v2/conv2/BatchNorm",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_3_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_3/bottleneck_v2/conv3",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block3_unit_4_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_4/bottleneck_v2/preact",
                num_features=1024,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_4_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_4/bottleneck_v2/conv1",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block3_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_4/bottleneck_v2/conv1/BatchNorm",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_4_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_4/bottleneck_v2/conv2",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block3_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_4/bottleneck_v2/conv2/BatchNorm",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_4_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_4/bottleneck_v2/conv3",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block3_unit_5_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_5/bottleneck_v2/preact",
                num_features=1024,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_5_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_5/bottleneck_v2/conv1",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block3_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_5/bottleneck_v2/conv1/BatchNorm",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_5_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_5/bottleneck_v2/conv2",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block3_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_5/bottleneck_v2/conv2/BatchNorm",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_5_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_5/bottleneck_v2/conv3",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block3_unit_6_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_6/bottleneck_v2/preact",
                num_features=1024,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_6_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_6/bottleneck_v2/conv1",
            in_channels=1024,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block3_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_6/bottleneck_v2/conv1/BatchNorm",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_6_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_6/bottleneck_v2/conv2",
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(2, 2),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block3_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block3/unit_6/bottleneck_v2/conv2/BatchNorm",
                num_features=256,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block3_unit_6_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block3/unit_6/bottleneck_v2/conv3",
            in_channels=256,
            out_channels=1024,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block4_unit_1_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block4/unit_1/bottleneck_v2/preact",
                num_features=1024,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block4_unit_1_bottleneck_v2_shortcut_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block4/unit_1/bottleneck_v2/shortcut",
            in_channels=1024,
            out_channels=2048,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block4_unit_1_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block4/unit_1/bottleneck_v2/conv1",
            in_channels=1024,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block4_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block4/unit_1/bottleneck_v2/conv1/BatchNorm",
                num_features=512,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block4_unit_1_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block4/unit_1/bottleneck_v2/conv2",
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block4_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block4/unit_1/bottleneck_v2/conv2/BatchNorm",
                num_features=512,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block4_unit_1_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block4/unit_1/bottleneck_v2/conv3",
            in_channels=512,
            out_channels=2048,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block4_unit_2_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block4/unit_2/bottleneck_v2/preact",
                num_features=2048,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block4_unit_2_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block4/unit_2/bottleneck_v2/conv1",
            in_channels=2048,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block4_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block4/unit_2/bottleneck_v2/conv1/BatchNorm",
                num_features=512,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block4_unit_2_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block4/unit_2/bottleneck_v2/conv2",
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block4_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block4/unit_2/bottleneck_v2/conv2/BatchNorm",
                num_features=512,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block4_unit_2_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block4/unit_2/bottleneck_v2/conv3",
            in_channels=512,
            out_channels=2048,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_block4_unit_3_bottleneck_v2_preact_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block4/unit_3/bottleneck_v2/preact",
                num_features=2048,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block4_unit_3_bottleneck_v2_conv1_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block4/unit_3/bottleneck_v2/conv1",
            in_channels=2048,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block4_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block4/unit_3/bottleneck_v2/conv1/BatchNorm",
                num_features=512,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block4_unit_3_bottleneck_v2_conv2_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block4/unit_3/bottleneck_v2/conv2",
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=1,
            bias=None,
        )
        self.resnet_v2_50_block4_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = (
            self.__batch_normalization(
                2,
                "resnet_v2_50/block4/unit_3/bottleneck_v2/conv2/BatchNorm",
                num_features=512,
                eps=1.0009999641624745e-05,
                momentum=0.0,
            )
        )
        self.resnet_v2_50_block4_unit_3_bottleneck_v2_conv3_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/block4/unit_3/bottleneck_v2/conv3",
            in_channels=512,
            out_channels=2048,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )
        self.resnet_v2_50_postnorm_FusedBatchNorm = self.__batch_normalization(
            2,
            "resnet_v2_50/postnorm",
            num_features=2048,
            eps=1.0009999641624745e-05,
            momentum=0.0,
        )
        self.resnet_v2_50_logits_Conv2D = self.__conv(
            2,
            name="resnet_v2_50/logits",
            in_channels=2048,
            out_channels=4096,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet feature extractor

        Parameters
        ----------
        x : torch.Tensor
            [B, C, H, W], range [0, 255]

        Returns
        -------
        torch.Tensor
            [B, 4096]
        """
        x = self.preprocess_img(x)

        resnet_v2_50_Pad = F.pad(x, (3, 3, 3, 3), mode="constant", value=0)
        resnet_v2_50_conv1_Conv2D = self.resnet_v2_50_conv1_Conv2D(resnet_v2_50_Pad)
        resnet_v2_50_pool1_MaxPool_pad = F.pad(
            resnet_v2_50_conv1_Conv2D, (0, 1, 0, 1), value=float("-inf")
        )
        resnet_v2_50_pool1_MaxPool, resnet_v2_50_pool1_MaxPool_idx = F.max_pool2d(
            resnet_v2_50_pool1_MaxPool_pad,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=0,
            ceil_mode=False,
            return_indices=True,
        )
        resnet_v2_50_block1_unit_1_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block1_unit_1_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_pool1_MaxPool
            )
        )
        resnet_v2_50_block1_unit_1_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block1_unit_1_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block1_unit_1_bottleneck_v2_shortcut_Conv2D = (
            self.resnet_v2_50_block1_unit_1_bottleneck_v2_shortcut_Conv2D(
                resnet_v2_50_block1_unit_1_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block1_unit_1_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block1_unit_1_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block1_unit_1_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block1_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block1_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block1_unit_1_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block1_unit_1_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block1_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block1_unit_1_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block1_unit_1_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block1_unit_1_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block1_unit_1_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block1_unit_1_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block1_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block1_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block1_unit_1_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block1_unit_1_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block1_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block1_unit_1_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block1_unit_1_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block1_unit_1_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block1_unit_1_bottleneck_v2_add = (
            resnet_v2_50_block1_unit_1_bottleneck_v2_shortcut_Conv2D
            + resnet_v2_50_block1_unit_1_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block1_unit_2_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block1_unit_2_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block1_unit_1_bottleneck_v2_add
            )
        )
        resnet_v2_50_block1_unit_2_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block1_unit_2_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block1_unit_2_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block1_unit_2_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block1_unit_2_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block1_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block1_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block1_unit_2_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block1_unit_2_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block1_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block1_unit_2_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block1_unit_2_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block1_unit_2_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block1_unit_2_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block1_unit_2_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block1_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block1_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block1_unit_2_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block1_unit_2_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block1_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block1_unit_2_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block1_unit_2_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block1_unit_2_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block1_unit_2_bottleneck_v2_add = (
            resnet_v2_50_block1_unit_1_bottleneck_v2_add
            + resnet_v2_50_block1_unit_2_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block1_unit_3_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block1_unit_3_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block1_unit_2_bottleneck_v2_add
            )
        )
        (
            resnet_v2_50_block1_unit_3_bottleneck_v2_shortcut_MaxPool,
            resnet_v2_50_block1_unit_3_bottleneck_v2_shortcut_MaxPool_idx,
        ) = F.max_pool2d(
            resnet_v2_50_block1_unit_2_bottleneck_v2_add,
            kernel_size=(1, 1),
            stride=(2, 2),
            padding=0,
            ceil_mode=False,
            return_indices=True,
        )
        resnet_v2_50_block1_unit_3_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block1_unit_3_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block1_unit_3_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block1_unit_3_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block1_unit_3_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block1_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block1_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block1_unit_3_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block1_unit_3_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block1_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block1_unit_3_bottleneck_v2_Pad = F.pad(
            resnet_v2_50_block1_unit_3_bottleneck_v2_conv1_Relu,
            (1, 1, 1, 1),
            mode="constant",
            value=0,
        )
        resnet_v2_50_block1_unit_3_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block1_unit_3_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block1_unit_3_bottleneck_v2_Pad
            )
        )
        resnet_v2_50_block1_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block1_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block1_unit_3_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block1_unit_3_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block1_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block1_unit_3_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block1_unit_3_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block1_unit_3_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block1_unit_3_bottleneck_v2_add = (
            resnet_v2_50_block1_unit_3_bottleneck_v2_shortcut_MaxPool
            + resnet_v2_50_block1_unit_3_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block2_unit_1_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block2_unit_1_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block1_unit_3_bottleneck_v2_add
            )
        )
        resnet_v2_50_block2_unit_1_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block2_unit_1_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block2_unit_1_bottleneck_v2_shortcut_Conv2D = (
            self.resnet_v2_50_block2_unit_1_bottleneck_v2_shortcut_Conv2D(
                resnet_v2_50_block2_unit_1_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block2_unit_1_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block2_unit_1_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block2_unit_1_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block2_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block2_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block2_unit_1_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block2_unit_1_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block2_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block2_unit_1_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block2_unit_1_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block2_unit_1_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block2_unit_1_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block2_unit_1_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block2_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block2_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block2_unit_1_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block2_unit_1_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block2_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block2_unit_1_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block2_unit_1_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block2_unit_1_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block2_unit_1_bottleneck_v2_add = (
            resnet_v2_50_block2_unit_1_bottleneck_v2_shortcut_Conv2D
            + resnet_v2_50_block2_unit_1_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block2_unit_2_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block2_unit_2_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block2_unit_1_bottleneck_v2_add
            )
        )
        resnet_v2_50_block2_unit_2_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block2_unit_2_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block2_unit_2_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block2_unit_2_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block2_unit_2_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block2_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block2_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block2_unit_2_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block2_unit_2_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block2_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block2_unit_2_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block2_unit_2_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block2_unit_2_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block2_unit_2_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block2_unit_2_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block2_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block2_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block2_unit_2_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block2_unit_2_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block2_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block2_unit_2_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block2_unit_2_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block2_unit_2_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block2_unit_2_bottleneck_v2_add = (
            resnet_v2_50_block2_unit_1_bottleneck_v2_add
            + resnet_v2_50_block2_unit_2_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block2_unit_3_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block2_unit_3_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block2_unit_2_bottleneck_v2_add
            )
        )
        resnet_v2_50_block2_unit_3_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block2_unit_3_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block2_unit_3_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block2_unit_3_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block2_unit_3_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block2_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block2_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block2_unit_3_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block2_unit_3_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block2_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block2_unit_3_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block2_unit_3_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block2_unit_3_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block2_unit_3_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block2_unit_3_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block2_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block2_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block2_unit_3_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block2_unit_3_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block2_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block2_unit_3_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block2_unit_3_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block2_unit_3_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block2_unit_3_bottleneck_v2_add = (
            resnet_v2_50_block2_unit_2_bottleneck_v2_add
            + resnet_v2_50_block2_unit_3_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block2_unit_4_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block2_unit_4_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block2_unit_3_bottleneck_v2_add
            )
        )
        (
            resnet_v2_50_block2_unit_4_bottleneck_v2_shortcut_MaxPool,
            resnet_v2_50_block2_unit_4_bottleneck_v2_shortcut_MaxPool_idx,
        ) = F.max_pool2d(
            resnet_v2_50_block2_unit_3_bottleneck_v2_add,
            kernel_size=(1, 1),
            stride=(2, 2),
            padding=0,
            ceil_mode=False,
            return_indices=True,
        )
        resnet_v2_50_block2_unit_4_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block2_unit_4_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block2_unit_4_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block2_unit_4_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block2_unit_4_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block2_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block2_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block2_unit_4_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block2_unit_4_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block2_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block2_unit_4_bottleneck_v2_Pad = F.pad(
            resnet_v2_50_block2_unit_4_bottleneck_v2_conv1_Relu,
            (1, 1, 1, 1),
            mode="constant",
            value=0,
        )
        resnet_v2_50_block2_unit_4_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block2_unit_4_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block2_unit_4_bottleneck_v2_Pad
            )
        )
        resnet_v2_50_block2_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block2_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block2_unit_4_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block2_unit_4_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block2_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block2_unit_4_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block2_unit_4_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block2_unit_4_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block2_unit_4_bottleneck_v2_add = (
            resnet_v2_50_block2_unit_4_bottleneck_v2_shortcut_MaxPool
            + resnet_v2_50_block2_unit_4_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block3_unit_1_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block3_unit_1_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block2_unit_4_bottleneck_v2_add
            )
        )
        resnet_v2_50_block3_unit_1_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block3_unit_1_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_1_bottleneck_v2_shortcut_Conv2D = (
            self.resnet_v2_50_block3_unit_1_bottleneck_v2_shortcut_Conv2D(
                resnet_v2_50_block3_unit_1_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block3_unit_1_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block3_unit_1_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block3_unit_1_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block3_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block3_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block3_unit_1_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block3_unit_1_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block3_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_1_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block3_unit_1_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block3_unit_1_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block3_unit_1_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block3_unit_1_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block3_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block3_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block3_unit_1_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block3_unit_1_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block3_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_1_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block3_unit_1_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block3_unit_1_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block3_unit_1_bottleneck_v2_add = (
            resnet_v2_50_block3_unit_1_bottleneck_v2_shortcut_Conv2D
            + resnet_v2_50_block3_unit_1_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block3_unit_2_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block3_unit_2_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block3_unit_1_bottleneck_v2_add
            )
        )
        resnet_v2_50_block3_unit_2_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block3_unit_2_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_2_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block3_unit_2_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block3_unit_2_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block3_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block3_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block3_unit_2_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block3_unit_2_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block3_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_2_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block3_unit_2_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block3_unit_2_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block3_unit_2_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block3_unit_2_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block3_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block3_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block3_unit_2_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block3_unit_2_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block3_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_2_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block3_unit_2_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block3_unit_2_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block3_unit_2_bottleneck_v2_add = (
            resnet_v2_50_block3_unit_1_bottleneck_v2_add
            + resnet_v2_50_block3_unit_2_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block3_unit_3_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block3_unit_3_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block3_unit_2_bottleneck_v2_add
            )
        )
        resnet_v2_50_block3_unit_3_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block3_unit_3_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_3_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block3_unit_3_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block3_unit_3_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block3_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block3_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block3_unit_3_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block3_unit_3_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block3_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_3_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block3_unit_3_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block3_unit_3_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block3_unit_3_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block3_unit_3_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block3_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block3_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block3_unit_3_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block3_unit_3_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block3_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_3_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block3_unit_3_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block3_unit_3_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block3_unit_3_bottleneck_v2_add = (
            resnet_v2_50_block3_unit_2_bottleneck_v2_add
            + resnet_v2_50_block3_unit_3_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block3_unit_4_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block3_unit_4_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block3_unit_3_bottleneck_v2_add
            )
        )
        resnet_v2_50_block3_unit_4_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block3_unit_4_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_4_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block3_unit_4_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block3_unit_4_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block3_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block3_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block3_unit_4_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block3_unit_4_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block3_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_4_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block3_unit_4_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block3_unit_4_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block3_unit_4_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block3_unit_4_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block3_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block3_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block3_unit_4_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block3_unit_4_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block3_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_4_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block3_unit_4_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block3_unit_4_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block3_unit_4_bottleneck_v2_add = (
            resnet_v2_50_block3_unit_3_bottleneck_v2_add
            + resnet_v2_50_block3_unit_4_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block3_unit_5_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block3_unit_5_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block3_unit_4_bottleneck_v2_add
            )
        )
        resnet_v2_50_block3_unit_5_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block3_unit_5_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_5_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block3_unit_5_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block3_unit_5_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block3_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block3_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block3_unit_5_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block3_unit_5_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block3_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_5_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block3_unit_5_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block3_unit_5_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block3_unit_5_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block3_unit_5_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block3_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block3_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block3_unit_5_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block3_unit_5_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block3_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_5_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block3_unit_5_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block3_unit_5_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block3_unit_5_bottleneck_v2_add = (
            resnet_v2_50_block3_unit_4_bottleneck_v2_add
            + resnet_v2_50_block3_unit_5_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block3_unit_6_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block3_unit_6_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block3_unit_5_bottleneck_v2_add
            )
        )
        (
            resnet_v2_50_block3_unit_6_bottleneck_v2_shortcut_MaxPool,
            resnet_v2_50_block3_unit_6_bottleneck_v2_shortcut_MaxPool_idx,
        ) = F.max_pool2d(
            resnet_v2_50_block3_unit_5_bottleneck_v2_add,
            kernel_size=(1, 1),
            stride=(2, 2),
            padding=0,
            ceil_mode=False,
            return_indices=True,
        )
        resnet_v2_50_block3_unit_6_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block3_unit_6_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_6_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block3_unit_6_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block3_unit_6_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block3_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block3_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block3_unit_6_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block3_unit_6_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block3_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_6_bottleneck_v2_Pad = F.pad(
            resnet_v2_50_block3_unit_6_bottleneck_v2_conv1_Relu,
            (1, 1, 1, 1),
            mode="constant",
            value=0,
        )
        resnet_v2_50_block3_unit_6_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block3_unit_6_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block3_unit_6_bottleneck_v2_Pad
            )
        )
        resnet_v2_50_block3_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block3_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block3_unit_6_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block3_unit_6_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block3_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block3_unit_6_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block3_unit_6_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block3_unit_6_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block3_unit_6_bottleneck_v2_add = (
            resnet_v2_50_block3_unit_6_bottleneck_v2_shortcut_MaxPool
            + resnet_v2_50_block3_unit_6_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block4_unit_1_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block4_unit_1_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block3_unit_6_bottleneck_v2_add
            )
        )
        resnet_v2_50_block4_unit_1_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block4_unit_1_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block4_unit_1_bottleneck_v2_shortcut_Conv2D = (
            self.resnet_v2_50_block4_unit_1_bottleneck_v2_shortcut_Conv2D(
                resnet_v2_50_block4_unit_1_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block4_unit_1_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block4_unit_1_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block4_unit_1_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block4_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block4_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block4_unit_1_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block4_unit_1_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block4_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block4_unit_1_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block4_unit_1_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block4_unit_1_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block4_unit_1_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block4_unit_1_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block4_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block4_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block4_unit_1_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block4_unit_1_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block4_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block4_unit_1_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block4_unit_1_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block4_unit_1_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block4_unit_1_bottleneck_v2_add = (
            resnet_v2_50_block4_unit_1_bottleneck_v2_shortcut_Conv2D
            + resnet_v2_50_block4_unit_1_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block4_unit_2_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block4_unit_2_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block4_unit_1_bottleneck_v2_add
            )
        )
        resnet_v2_50_block4_unit_2_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block4_unit_2_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block4_unit_2_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block4_unit_2_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block4_unit_2_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block4_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block4_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block4_unit_2_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block4_unit_2_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block4_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block4_unit_2_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block4_unit_2_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block4_unit_2_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block4_unit_2_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block4_unit_2_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block4_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block4_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block4_unit_2_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block4_unit_2_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block4_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block4_unit_2_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block4_unit_2_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block4_unit_2_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block4_unit_2_bottleneck_v2_add = (
            resnet_v2_50_block4_unit_1_bottleneck_v2_add
            + resnet_v2_50_block4_unit_2_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_block4_unit_3_bottleneck_v2_preact_FusedBatchNorm = (
            self.resnet_v2_50_block4_unit_3_bottleneck_v2_preact_FusedBatchNorm(
                resnet_v2_50_block4_unit_2_bottleneck_v2_add
            )
        )
        resnet_v2_50_block4_unit_3_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_50_block4_unit_3_bottleneck_v2_preact_FusedBatchNorm
        )
        resnet_v2_50_block4_unit_3_bottleneck_v2_conv1_Conv2D = (
            self.resnet_v2_50_block4_unit_3_bottleneck_v2_conv1_Conv2D(
                resnet_v2_50_block4_unit_3_bottleneck_v2_preact_Relu
            )
        )
        resnet_v2_50_block4_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block4_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block4_unit_3_bottleneck_v2_conv1_Conv2D
        )
        resnet_v2_50_block4_unit_3_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_50_block4_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block4_unit_3_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_50_block4_unit_3_bottleneck_v2_conv1_Relu, (1, 1, 1, 1)
        )
        resnet_v2_50_block4_unit_3_bottleneck_v2_conv2_Conv2D = (
            self.resnet_v2_50_block4_unit_3_bottleneck_v2_conv2_Conv2D(
                resnet_v2_50_block4_unit_3_bottleneck_v2_conv2_Conv2D_pad
            )
        )
        resnet_v2_50_block4_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_50_block4_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_50_block4_unit_3_bottleneck_v2_conv2_Conv2D
        )
        resnet_v2_50_block4_unit_3_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_50_block4_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm
        )
        resnet_v2_50_block4_unit_3_bottleneck_v2_conv3_Conv2D = (
            self.resnet_v2_50_block4_unit_3_bottleneck_v2_conv3_Conv2D(
                resnet_v2_50_block4_unit_3_bottleneck_v2_conv2_Relu
            )
        )
        resnet_v2_50_block4_unit_3_bottleneck_v2_add = (
            resnet_v2_50_block4_unit_2_bottleneck_v2_add
            + resnet_v2_50_block4_unit_3_bottleneck_v2_conv3_Conv2D
        )
        resnet_v2_50_postnorm_FusedBatchNorm = (
            self.resnet_v2_50_postnorm_FusedBatchNorm(
                resnet_v2_50_block4_unit_3_bottleneck_v2_add
            )
        )
        resnet_v2_50_postnorm_Relu = F.relu(resnet_v2_50_postnorm_FusedBatchNorm)
        resnet_v2_50_pool5 = torch.mean(resnet_v2_50_postnorm_Relu, 3, True)
        resnet_v2_50_pool5 = torch.mean(resnet_v2_50_pool5, 2, True)
        resnet_v2_50_logits_Conv2D = self.resnet_v2_50_logits_Conv2D(resnet_v2_50_pool5)
        MMdnn_Output = torch.squeeze(resnet_v2_50_logits_Conv2D)
        return MMdnn_Output

    def predict_exif(self, x1, x2):
        # x1, x2: [C, H, W]
        if len(x1.shape) == 3:
            x1 = x1.unsqueeze(0)
        if len(x2.shape) == 3:
            x2 = x2.unsqueeze(0)

        feat_1 = self(x1)
        feat_2 = self(x2)
        feat = torch.cat([feat_1, feat_2], dim=-1)

        logits = self.exif_fc(feat)

        # FIXME Sigmoid or nay?
        return logits
        # return torch.sigmoid(logits)  # [B, 83]

    def predict(self, x1, x2):
        exif_preds = self.predict_exif(x1, x2)
        logits = self.classifier_fc(exif_preds)

        return torch.sigmoid(logits)  # [B, 1]

    @staticmethod
    def __set_fc_params(module, name):
        if _weights_dict:
            module.state_dict()["weight"].copy_(
                torch.from_numpy(_weights_dict[name]["weights"])
            )
            module.state_dict()["bias"].copy_(
                torch.from_numpy(_weights_dict[name]["biases"])
            )

    @staticmethod
    def __conv(dim, name, **kwargs):
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError()

        if _weights_dict:
            layer.state_dict()["weight"].copy_(
                torch.from_numpy(_weights_dict[name]["weights"])
            )
            if "biases" in _weights_dict[name]:
                layer.state_dict()["bias"].copy_(
                    torch.from_numpy(_weights_dict[name]["biases"])
                )
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if dim == 0 or dim == 1:
            layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:
            layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:
            layer = nn.BatchNorm3d(**kwargs)
        else:
            raise NotImplementedError()

        if _weights_dict:
            if "gamma" in _weights_dict[name]:
                layer.state_dict()["weight"].copy_(
                    torch.from_numpy(_weights_dict[name]["gamma"])
                )
            else:
                layer.weight.data.fill_(1)

            if "beta" in _weights_dict[name]:
                layer.state_dict()["bias"].copy_(
                    torch.from_numpy(_weights_dict[name]["beta"])
                )
            else:
                layer.bias.data.fill_(0)

            layer.state_dict()["running_mean"].copy_(
                torch.from_numpy(_weights_dict[name]["moving_mean"])
            )
            layer.state_dict()["running_var"].copy_(
                torch.from_numpy(_weights_dict[name]["moving_variance"])
            )
        else:
            layer.weight.data.fill_(1)
            layer.bias.data.fill_(0)

        return layer
