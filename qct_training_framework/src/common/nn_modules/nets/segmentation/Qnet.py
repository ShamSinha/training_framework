from typing import List

import torch
import torch.nn as nn

from ..utils.convert_3d import convert_3d


class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1, 1)
        x = inputs * x
        return x


class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y


class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y


class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_c),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_c),
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_c),
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm2d(out_c),
        )

        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)
        return y


class Attention_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(),
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1),
            nn.MaxPool3d((2, 2, 2)),
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y


class Encoder(nn.Module):
    def __init__(self, encoder_c: List[int] = [1, 16, 32, 64, 128, 256]):
        super().__init__()
        self.c1 = Stem_Block(encoder_c[0], encoder_c[1], stride=1)
        self.c2 = ResNet_Block(encoder_c[1], encoder_c[2], stride=2)
        self.c3 = ResNet_Block(encoder_c[2], encoder_c[3], stride=2)
        self.c4 = ResNet_Block(encoder_c[3], encoder_c[4], stride=2)
        self.b1 = ASPP(encoder_c[4], encoder_c[5])

    def forward(self, x: torch.Tensor):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        b1 = self.b1(c4)
        return [c1, c2, c3, b1]


class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = Attention_Block(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNet_Block(in_c[0] + in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d


class Decoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int] = [1, 16, 32, 64, 128, 256],
        decoder_channels: List[int] = [128, 64, 32, 16],
    ):
        super().__init__()
        self.d1 = convert_3d(
            Decoder_Block([encoder_channels[3], encoder_channels[5]], decoder_channels[0])
        )
        self.d2 = convert_3d(
            Decoder_Block([encoder_channels[2], decoder_channels[0]], decoder_channels[1])
        )
        self.d3 = convert_3d(
            Decoder_Block([encoder_channels[1], decoder_channels[1]], decoder_channels[2])
        )
        self.aspp = convert_3d(ASPP(decoder_channels[2], decoder_channels[3]))

    def forward(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, b1: torch.Tensor):
        d1 = self.d1(c3, b1)
        d2 = self.d2(c2, d1)
        d3 = self.d3(c1, d2)
        output = self.aspp(d3)
        return output


class Qnet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        encoder_channels: List[int] = [1, 16, 32, 64, 128, 256],
        decoder_channels: List[int] = [128, 64, 32, 16],
    ):
        """Merging all the new things into Unet.

        Attention. ASPP. SELayer. ROI Encoder.
        With the default config model should have 16.42M parameters.
        Example:
        ```
        model = Qnet(num_classes=2)
        crop_batch = torch.randn(2, 1, 32, 64, 64)
        roi_batch = (torch.randn(2, 1, 32, 64, 64)>0.5).float()
        op = model(crop_batch, roi_batch)
        op.shape
        ```
        """
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes
        self.build_model()

    def build_model(self):
        self.encoder1 = convert_3d(Encoder(self.encoder_channels))
        self.encoder2 = convert_3d(Encoder(self.encoder_channels))
        se_layers_channels = self.decoder_channels[::-1]
        se_layers_channels[-1] *= 2
        self.se_layers = nn.ModuleList(
            [Squeeze_Excitation(chns, 4) for chns in se_layers_channels]
        )
        self.decoder = convert_3d(
            Decoder([_ for _ in self.encoder_channels], self.decoder_channels)
        )
        self.segmentation_head = convert_3d(
            nn.Conv2d(
                self.decoder_channels[-1],
                self.num_classes,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
        )

    def forward(self, images: torch.Tensor, roi_mask: torch.Tensor):
        global_features0 = self.encoder1(images)
        global_features1 = self.encoder2(roi_mask)
        global_features = [
            self.se_layers[i](global_feat0 + global_feat1)
            for i, (global_feat0, global_feat1) in enumerate(
                zip(global_features0, global_features1)
            )
        ]

        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features
