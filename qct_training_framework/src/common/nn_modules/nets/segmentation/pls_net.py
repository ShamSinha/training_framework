import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from typing import List

import torch.nn as nn
import torch
import torch.utils.checkpoint as cp


class DSConv3D(nn.Module):
    def __init__(self, in_chans, out_chans, dilation=1, dstride=2, padding=1):
        super(DSConv3D, self).__init__()
        self.dConv = nn.Conv3d(in_chans, in_chans, kernel_size=3, stride=dstride, padding=padding,
                               dilation=dilation, groups=in_chans, bias=False)
        self.conv = nn.Conv3d(in_chans, out_chans, kernel_size=1, dilation=1, stride=1, bias=False)
        self.norm = nn.BatchNorm3d(out_chans, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dConv(x)
        out = self.conv(out)
        out = self.relu(out)
        return out
    
    
class DrdbBlock3D(nn.Module):
    def __init__(self, in_chans, out_chans, growth_rate, nr_blocks=4):
        super(DrdbBlock3D, self).__init__()
        self.nr_blocks = nr_blocks
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.growth_rate = growth_rate
        self.memory_efficient = True

        self.ds_conv_1 = DSConv3D(in_chans=self.in_chans, out_chans=growth_rate, dilation=1, dstride=1,
                                  padding=1)
        self.ds_conv_2 = DSConv3D(in_chans=self.in_chans + growth_rate, out_chans=growth_rate, dilation=2,
                                  dstride=1, padding=2)
        self.ds_conv_3 = DSConv3D(in_chans=self.in_chans + growth_rate * 2, out_chans=growth_rate,
                                  dilation=3, dstride=1, padding=3)
        self.ds_conv_4 = DSConv3D(in_chans=self.in_chans + growth_rate * 3, out_chans=growth_rate,
                                  dilation=4, dstride=1, padding=4)

        self.conv = nn.Conv3d(in_chans + growth_rate * 4, self.out_chans, kernel_size=1)

    def forward(self, x):
        if self.memory_efficient:
            out = cp.checkpoint(self.bottleneck_function, x)
        else:
            out = self.bottleneck_function(x)
        return out

    def bottleneck_function(self, x):
        out = self.ds_conv_1(x)
        cat = torch.cat([out, x], 1)
        out = self.ds_conv_2(cat)
        cat = torch.cat([out, cat], 1)
        out = self.ds_conv_3(cat)
        cat = torch.cat([out, cat], 1)
        out = self.ds_conv_4(cat)
        cat = torch.cat([out, cat], 1)
        out = self.conv(cat)
        out = torch.add(out, x)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(DecoderBlock, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.ds_conv = DSConv3D(in_chans=in_chans, out_chans=out_chans, dilation=1, dstride=1)
        self.upsampled = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x):
        out = self.ds_conv(x)
        out = self.upsampled(out)
        return out
    
class pls_encoder(nn.Module):
    def __init__(self,
                  nb_channels: List[int],
                  growth_rate: int
                ):
        super().__init__()

        self.nb_channels = nb_channels
        self.growth_rate = growth_rate

        self.ds_conv_1 = DSConv3D(self.nb_channels[0] + 1, self.nb_channels[1])
        self.drdb_1 = DrdbBlock3D(self.nb_channels[1] + 1, self.nb_channels[1] + 1, self.growth_rate)

        self.ds_conv_2 = DSConv3D(self.nb_channels[1] + 1, self.nb_channels[2])
        self.drdb_2_1 = DrdbBlock3D(self.nb_channels[2] + 1, self.nb_channels[2] + 1, self.growth_rate)
        self.drdb_2_2 = DrdbBlock3D(self.nb_channels[2] + 1, self.nb_channels[2] + 1, self.growth_rate)

        self.ds_conv_3 = DSConv3D(self.nb_channels[2] + 1, self.nb_channels[3])
        self.drdb_3_1 = DrdbBlock3D(self.nb_channels[3] + 1, self.nb_channels[3] + 1, self.growth_rate)
        self.drdb_3_2 = DrdbBlock3D(self.nb_channels[3] + 1, self.nb_channels[3] + 1, self.growth_rate)
        self.drdb_3_3 = DrdbBlock3D(self.nb_channels[3] + 1, self.nb_channels[3] + 1, self.growth_rate)
        self.drdb_3_4 = DrdbBlock3D(self.nb_channels[3] + 1, self.nb_channels[3] + 1, self.growth_rate)

    def forward(self, x):
        x = x
        input_ = x
        out = self.ds_conv_1(x)
        downsampled_1 = F.interpolate(input_, scale_factor=0.5, mode='trilinear', align_corners=False)
        out = torch.cat([out, downsampled_1], 1)
        out_l1 = self.drdb_1(out)

        # l = 2
        out = self.ds_conv_2(out_l1)
        downsampled_2 = F.interpolate(input_, scale_factor=0.25, mode='trilinear', align_corners=False)
        out = torch.cat([out, downsampled_2], 1)
        out = self.drdb_2_1(out)
        out_l2 = self.drdb_2_2(out)

        # l = 3
        out = self.ds_conv_3(out_l2)
        downsampled_3 = F.interpolate(input_, scale_factor=0.125, mode='trilinear', align_corners=False)
        out = torch.cat([out, downsampled_3], 1)
        out = self.drdb_3_1(out)
        out = self.drdb_3_2(out)
        out = self.drdb_3_3(out)
        out = self.drdb_3_4(out)

        return [out, out_l2, out_l1]

class pls_decoder(nn.Module):
    def __init__(self,
                nb_classes: int,
                nb_channels: List[int],
                ):
        super().__init__()

        self.nb_classes = nb_classes
        self.nb_channels = nb_channels

        self.ds_bridge_l2 = DSConv3D(in_chans=self.nb_channels[2] + 1, out_chans=self.nb_classes * 2, dstride=1)
        self.ds_bridge_l1 = DSConv3D(in_chans=self.nb_channels[1] + 1, out_chans=self.nb_classes * 2, dstride=1)

        self.decoder_l3 = DecoderBlock(in_chans=self.nb_channels[-1] + 1, out_chans=self.nb_classes * 2)
        self.decoder_l2 = DecoderBlock(in_chans=self.nb_classes * 4, out_chans=self.nb_classes * 2)
        self.decoder_l1 = DecoderBlock(in_chans=self.nb_classes * 4, out_chans=self.nb_classes * 2)

    def forward(self, encoder_out):
        out = self.decoder_l3(encoder_out[0])
        out = torch.cat([out, self.ds_bridge_l2(encoder_out[1])], 1)
        out = self.decoder_l2(out)
        out = torch.cat([out, self.ds_bridge_l1(encoder_out[2])], 1)
        out = self.decoder_l1(out)
        return out
    

class PLS(nn.Module):
    def __init__(self, nb_classes: int,
                  nb_channels: int,
                  growth_rate: int):
        super().__init__()
        self.nb_classes = nb_classes

        # Network specific arguments
        self.nb_channels = nb_channels
        self.growth_rate = growth_rate

        # ENCODER
        self.encoder = pls_encoder(nb_channels= self.nb_channels, growth_rate= self.growth_rate)
        # DECODER
        self.decoder = pls_decoder(nb_classes= self.nb_classes, nb_channels=self.nb_channels)
        # OUTPUT
        self.segmentation_head = nn.Conv3d(in_channels=self.nb_classes * 2, out_channels=self.nb_classes, kernel_size=1)


    def forward(self, x):
        features = self.encoder(x)
        decoder_out = self.decoder.forward(features)
        out = self.segmentation_head(decoder_out)
        return out