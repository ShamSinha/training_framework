import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from typing import Union, List

from ..utils.convert_3d import convert_3d
from loguru import logger


class TimmSegModel(nn.Module):
    def __init__(
        self,
        num_classes: Union[List[int] , int],
        label_keys: Union[List[str], str],
        n_blocks: int,
        backbone: str,
        segtype="unet",
        spatial_dims: int = 3,
        pretrained=False,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        in_channels: int = 1,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        attention_type = "scse"
    ):
        super().__init__()

        assert len(decoder_channels) >= n_blocks, "decoder_channels must be >= n_blocks"
        if isinstance(num_classes, list) and isinstance(label_keys, list) : 
            assert len(num_classes) == len(label_keys)

        self.label_keys = label_keys

        self.encoder = timm.create_model(
            backbone,
            in_chans = in_channels,
            features_only=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
        )
        self.n_blocks = n_blocks
        g = self.encoder(torch.rand(1, in_channels, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        if segtype == "unet":
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[: n_blocks + 1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )
        elif segtype == "unet++":
            self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
                encoder_channels=encoder_channels[: n_blocks + 1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
                use_batchnorm=True,
                attention_type= attention_type,
            )

        if isinstance(num_classes, int):
            self.segmentation_head = nn.Conv2d(
                decoder_channels[n_blocks - 1],
                num_classes,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
        else:
            self.segmentation_head = nn.ModuleList(
                [
                    nn.Conv2d(
                        decoder_channels[n_blocks - 1],
                        num_classes[i],
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    )
                    for i in range(len(num_classes))
                ]
            )

    def forward(self, x):
        global_features = [0] + self.encoder(x)[: self.n_blocks]
        seg_features = self.decoder(*global_features)

        output = {}
        if isinstance(self.segmentation_head, nn.ModuleList):
            for i in range(len(self.segmentation_head)):
                output[self.label_keys[i]] = self.segmentation_head[i](seg_features)
        else:
            output = self.segmentation_head(seg_features)

        return output
