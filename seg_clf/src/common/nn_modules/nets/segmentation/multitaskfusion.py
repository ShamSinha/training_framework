import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.base import ClassificationHead
from copy import deepcopy
from segmentation_models_pytorch.base import initialization as init
from loguru import logger

class MultiTaskFusionNet(torch.nn.Module):
    def __init__(
        self,
        encoder_name="resnet18",
        architecture="Unet",
        num_classes=2,
        encoder_depth=5,
        encoder_weights = "imagenet",
        decoder_attention_type = "scse"
    ):      
        super().__init__()
        
        self.teacher_backbone = getattr(smp, "Unet")(
            encoder_name="resnet18",
            encoder_depth=5,
            classes= 2,
            encoder_weights= None,
            decoder_attention_type="gated_sse",
        )

        checkpoint_path = "/home/users/shubham.kumar/projects/qureai/packages/python/qer/resources/checkpoints/hemorrhages_quantification/FDA_approved_mixed_mask.pth"
        kwargs = {}
        kwargs["map_location"] = "cpu"
        checkpoint = torch.load(checkpoint_path, **kwargs)

        ###################
        backbone_checkpoint = deepcopy(checkpoint)
        ####################

        for key in ["win_opt.conv2d.weight", "win_opt.conv2d.bias", "multi_fc.fc_0.weight", "multi_fc.fc_0.bias"]:
            del backbone_checkpoint["model_state_dict"][key]
        new_state_dict = modify_model_state_dict(backbone_checkpoint["model_state_dict"], "backbone.encoder" , "encoder")
        new_state_dict = modify_model_state_dict(new_state_dict, "backbone.decoder" , "decoder")
        new_state_dict = modify_model_state_dict(new_state_dict, "backbone.segmentation_head" , "segmentation_head")

        self.teacher_backbone.load_state_dict(new_state_dict)

        # self.student_backbone = getattr(smp, architecture)(
        #     encoder_name = encoder_name,
        #     encoder_depth=encoder_depth,
        #     classes= num_classes,
        #     encoder_weights= encoder_weights,
        #     decoder_attention_type=decoder_attention_type,
        # )

        self.student_backbone = deepcopy(self.teacher_backbone)

    def initialize(self):
        init.initialize_decoder(self.backbone.decoder)

    def forward(self, x):

        # x = (b, 3, z, 224, 224)
        z_size = x.size(2)
        # fold z in to batch dimension
        x = x.transpose(1, 2).contiguous()       # (b, z, 3, 224, 224)
        x = x.view(-1, *x.size()[2:])            # (b*z, 3, 224, 224)

        outputs = {}

        student_features = self.student_backbone.encoder(x)
        decoder_output = self.student_backbone.decoder(*student_features)
        seg_output = self.student_backbone.segmentation_head(decoder_output)
        outputs["student_mask"] = torch.swapaxes(torch.stack(torch.split(seg_output, z_size)), 1 ,2)

        with torch.no_grad():
            teacher_features = self.teacher_backbone.encoder(x)
            decoder_output = self.teacher_backbone.decoder(*teacher_features)
            seg_output = self.teacher_backbone.segmentation_head(decoder_output)[:,0,:,:]
            seg_output = torch.stack([-seg_output,seg_output],dim=1)
            outputs["teacher_mask"] = torch.swapaxes(torch.stack(torch.split(seg_output, z_size)), 1 ,2)

        return outputs
    
def modify_model_state_dict(model_state_dict, old_prefix, new_prefix, modify_conv1x1=False):
    """Modify the prefixes in model_state_dict."""
    model_state_dict = {k.replace(old_prefix, new_prefix): v for k, v in model_state_dict.items()}

    if modify_conv1x1:
        conv1x1_keys = [k for k in model_state_dict.keys() if "multi_conv.conv1x1" in k]
        for k in conv1x1_keys:
            k_new = k.replace("multi_conv.conv1x1", "multi_fc.fc")
            model_state_dict[k_new] = model_state_dict[k].squeeze()
            del model_state_dict[k]

    return model_state_dict