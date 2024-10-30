from torch import nn
import torch
from loguru import logger

def modify_model_state_dict(
    model_state_dict, old_prefix, new_prefix, modify_conv1x1=False
):
    """Modify the prefixes in model_state_dict."""
    model_state_dict = {
        k.replace(old_prefix, new_prefix): v for k, v in model_state_dict.items()
    }

    if modify_conv1x1:
        conv1x1_keys = [k for k in model_state_dict.keys() if "multi_conv.conv1x1" in k]
        for k in conv1x1_keys:
            k_new = k.replace("multi_conv.conv1x1", "multi_fc.fc")
            model_state_dict[k_new] = model_state_dict[k].squeeze()
            del model_state_dict[k]

    return model_state_dict


class ViTMAEForImageClassification(nn.Module):
    def __init__(self, model_ckpt, model):
        super().__init__()
        
        self.model = model
        checkpoint = torch.load(model_ckpt, map_location = "cpu")

        state_dict = checkpoint["state_dict"]
        new_state_dict = modify_model_state_dict(state_dict, "model.", "")

        self.model.load_state_dict(new_state_dict)

        del self.model.decoder
        del self.model.encoder_to_decoder
        del self.model.criterion

        # Classifier head
        self.classifier = nn.Linear(self.model.encoder_embed_dim, 2)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        image = x.permute(0, 1, 3, 4, 2)
        classification_embeddings = self.model.get_classification_embed(image)
        logits = self.classifier(classification_embeddings)
        return logits



