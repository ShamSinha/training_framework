import torch
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init

from torchvision.ops import SqueezeExcitation


class UnetPlusPlus(nn.Module):
    def __init__(self, 
                 encoder_name = "resnet18",
                 in_channels = 1,
                 num_classes: int = 1,
                 decoder_channels = [64, 32, 16],
                 encoder_depth = 3,
                 decoder_use_batchnorm = True,
                 decoder_attention_type = None,):
        super().__init__()

        self.model = smp.UnetPlusPlus(encoder_name=encoder_name,
                                      in_channels= in_channels,
                                    
                                      classes= num_classes,
                                      decoder_channels= decoder_channels,
                                      encoder_depth= encoder_depth, 
                                      encoder_weights='imagenet',
                                      decoder_use_batchnorm= decoder_use_batchnorm,
                                      decoder_attention_type= decoder_attention_type)

    def forward(self, ct):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        masks = self.model(ct)
        return masks
    

class Unet_Modified(nn.Module):
    def __init__(self, 
                 encoder_name = "resnet18",
                 in_channels = 1,
                 out_channels: int = 1,
                 decoder_channels = [64, 32, 16],
                 encoder_depth = 3,
                 decoder_use_batchnorm = True,
                 decoder_attention_type = None,
                 seg_kernel_size = 3):
        super().__init__()
        self.segmentation_head = None

        self.encoder = smp.encoders.get_encoder(name= encoder_name,
                                                in_channels= in_channels,
                                                depth= encoder_depth, 
                                                weights='imagenet')

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels= decoder_channels,
            n_blocks= encoder_depth,
            use_batchnorm= decoder_use_batchnorm,
            attention_type= decoder_attention_type,
            center=True if encoder_name.startswith("vgg") else False,
        )
        self.segmentation_head = SegmentationHead(
            in_channels= decoder_channels[-1],
            out_channels= out_channels,
            activation = None ,
            kernel_size = seg_kernel_size,
        )

        self.initialize()

    def initialize(self):
        if self.segmentation_head is not None:
            init.initialize_decoder(self.decoder)
            init.initialize_head(self.segmentation_head)

    def forward(self, ct):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        
        masks = None
        features = self.encoder(ct)
        
        if self.segmentation_head is not None:
            decoder_output = self.decoder(*features)
            masks = self.segmentation_head(decoder_output)

        return masks