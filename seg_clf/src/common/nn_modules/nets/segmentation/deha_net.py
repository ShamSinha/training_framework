import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn

from ..utils.convert_3d import convert_3d


class DEHANET(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_blocks: int,
        in_chans: int,
        encoder_backbone: str,
        segtype: str = "unet",
        drop_rate: float = 0,
        drop_path_rate: float = 0,
        pretrained: bool = False,
    ):
        """
        Implementation of DEHA Net. Detailed description of the paper https://github.com/qureai/qct_training_framework/issues/57
        Args:
        -----
        num_classes: number of segmentation classes including the background
        n_blocks: number of encoder blocks to extract the features from.
        in_chans: input channels in the data
        encoder_backbone: encoder backbone name. use one of the Timm backbones.
        segtype: decoder type. Currently only supports Unet and UnetPlusPlus.
        drop_rate: drop rate
        drop_path_rate: drop path
        pretrained: encoder uses the imagenet pretrained weights

        ```
        import torch
        from src.common.nn_modules.nets.segmentation.deha_net import DEHANET
        crop_batch = torch.randn(10, 1, 32, 64, 64)
        roi_batch = (torch.randn(10, 1, 32, 64, 64)>0.5).float()

        dehanet = DEHANET(num_classes=2, n_blocks=4, in_chans=1, encoder_backbone="resnet18", segtype="unet")
        model_op = dehanet([crop_batch, roi_batch])
        ```
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.encoder1 = convert_3d(
            timm.create_model(
                encoder_backbone,
                in_chans=in_chans,
                features_only=True,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                pretrained=pretrained,
            )
        )
        self.encoder2 = convert_3d(
            timm.create_model(
                encoder_backbone,
                in_chans=in_chans,
                features_only=True,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                pretrained=pretrained,
            )
        )
        with torch.no_grad():
            g = self.encoder1(torch.rand(1, 1, 64, 64, 64))
            encoder_channels = [1] + [_.shape[1] for _ in g]
        encoder_channels = [chns * 2 for chns in encoder_channels]
        decoder_channels = [256, 128, 64, 32, 16]

        if segtype == "unet":
            decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[: n_blocks + 1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )
        elif segtype == "unet++":
            decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
                encoder_channels=encoder_channels[: n_blocks + 1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
                use_batchnorm=True,
            )
        self.decoder = convert_3d(decoder)

        self.segmentation_head = convert_3d(
            nn.Conv2d(
                decoder_channels[n_blocks - 1],
                num_classes,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
        )

    def forward(self, images: torch.Tensor, roi_mask: torch.Tensor):
        global_features0 = self.encoder1(images)[: self.n_blocks]
        global_features1 = self.encoder2(roi_mask)[: self.n_blocks]
        global_features = [torch.tensor(0)] + [
            torch.cat([global_feat0, global_feat1], dim=1).to(global_features0[0].device)
            for global_feat0, global_feat1 in zip(global_features0, global_features1)
        ]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features
