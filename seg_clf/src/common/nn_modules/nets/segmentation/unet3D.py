import torch
import torch.nn as nn
import torch.nn.functional as F
import qure_segmentation_models_pytorch as qure_smp
from typing import List

from qure_segmentation_models_pytorch.encoders import get_encoder
from qure_segmentation_models_pytorch.base import SegmentationHead_3D
from qure_segmentation_models_pytorch.decoders.unet_3D import UnetDecoder_3D
from segmentation_models_pytorch.base import initialization as init

class SmpUnet3DModel(nn.Module):
    def __init__(
        self,
        num_classes: List[int],
        in_channels : int,
        n_blocks: int,
        backbone: str,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        use_batchnorm: bool = True,
        attention_type = "scse",

    ):
        super().__init__()

        self.n_blocks = n_blocks
        self.num_classes = num_classes

        self.encoder = get_encoder(
            encoder_name = backbone,
            in_channels=in_channels,
            depth= n_blocks,
            weights=None,
        )

        self.decoder = UnetDecoder_3D(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels[:n_blocks],
            n_blocks= n_blocks,
            use_batchnorm= use_batchnorm,
            center=True if backbone.startswith("vgg") else False,
            attention_type= attention_type,
        )

        self.segmentation_head_lobe = SegmentationHead_3D(
                in_channels=decoder_channels[-1],
                out_channels=  num_classes[0], 
                activation= None,
                kernel_size=3,
                temporal_size=1
            )

        self.segmentation_head_lung = SegmentationHead_3D(
                in_channels=decoder_channels[-1],
                out_channels= num_classes[1], 
                activation= None,
                kernel_size=3,
                temporal_size=1
            )

        self.initialize()

    def initialize(self):
        if self.segmentation_head is not None:
            init.initialize_decoder(self.decoder)

            init.initialize_head(self.segmentation_head_lobe)
            init.initialize_head(self.segmentation_head_lung)

    def forward(self, x):
        global_features = self.encoder(x)
        seg_features = self.decoder(*global_features)

        output = {}

        output["lung_label"] = self.segmentation_head_lung(seg_features)
        output["lobe_label"] = self.segmentation_head_lobe(seg_features)

        return output

