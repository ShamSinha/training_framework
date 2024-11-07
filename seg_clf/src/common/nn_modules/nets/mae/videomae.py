import torch
import lightning.pytorch as pl
from torch import nn

from mmcv import Config
from functools import partial
from transformers import VideoMAEConfig, VideoMAEForPreTraining

class VideoMAE(nn.Module):
    def __init__(self,
                 image_size : int = 32,
                 tubelet_size: int = 4,
                 num_channels: int = 1,
                 num_frames: int = 8):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = VideoMAEConfig(image_size=image_size, tubelet_size=tubelet_size, num_channels=num_channels, num_frames=num_frames,)
        self.model = VideoMAEForPreTraining(config=self.model_config)

    def forward(self,x) :
        outputs = self.model(x)
        return outputs.logits






