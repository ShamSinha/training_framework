import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining

# Step 1: Define your ViT model
class CustomViT(nn.Module):
    def __init__(self):
        # configuration = ViTMAEConfig()
        # model = ViTMAEModel(configuration)
        # self.model  = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        self.model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        
    def forward(self, x):
        outputs = self.model(x)
        pass

        # return outputs.last_hidden_state
