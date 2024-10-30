import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.base import ClassificationHead
from copy import deepcopy
from segmentation_models_pytorch.base import initialization as init
from loguru import logger

class MultiTaskNet(torch.nn.Module):
    def __init__(
        self,
        cls_pooling = "avg",
        cls_dropout = 0.5,
        decoder_attention_type="gated_sse",
    ):      
        super().__init__()
        
        encoder_name="resnet18"
        architecture="Unet"
        in_chans = 3
        num_classes=2
        encoder_depth=5
        
        self.backbone = getattr(smp, architecture)(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            classes= num_classes,
            encoder_weights= "imagenet",
            decoder_attention_type=decoder_attention_type,
        )
        # self.init()

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

        self.backbone.load_state_dict(new_state_dict)


        # for key in checkpoint["model_state_dict"].keys() :
        #     if key not in ["multi_fc.fc_0.weight", "multi_fc.fc_0.bias"] :
        #         del cls_head_checkpoint["model_state_dict"][key]

        # cls_head_checkpoint["model_state_dict"]["3.weight"] = cls_head_checkpoint["model_state_dict"]["multi_fc.fc_0.weight"]
        # cls_head_checkpoint["model_state_dict"]["3.bias"] = cls_head_checkpoint["model_state_dict"]["multi_fc.fc_0.bias"]

        # for key in ["multi_fc.fc_0.weight", "multi_fc.fc_0.bias"] :
        #     del cls_head_checkpoint["model_state_dict"][key]

        #########################


        # self.classification_head = ClassificationHead(
        #     in_channels=528,
        #     classes= num_classes,
        #     pooling= cls_pooling,
        #     dropout= cls_dropout,
        #     activation=None,
        # )

        # self.classification_head.load_state_dict(cls_head_checkpoint["model_state_dict"])


    # def init(self, expected_fraction=0.99):
    #     """Initialize model parameters.

    #     Initialize segmentation head so that background is predicted
    #     most of the time. This idea is from Focal Loss paper.
    #     """
    #     fill_0 = np.log(expected_fraction / (1 - expected_fraction)) / 2
    #     bias = self.backbone.segmentation_head[0].bias.data
    #     bias[0].fill_(fill_0)
    #     bias[1].fill_(-fill_0)

    def initialize(self):
        init.initialize_decoder(self.backbone.decoder)
        # init.initialize_head(self.classification_head)

    def forward(self, x):

        # x = (b, 3, z, 224, 224)
        z_size = x.size(2)
        # fold z in to batch dimension
        x = x.transpose(1, 2).contiguous()       # (b, z, 3, 224, 224)
        x = x.view(-1, *x.size()[2:])            # (b*z, 3, 224, 224)

        features = self.backbone.encoder(x)

        # classification_out = self.classification_head(features[-1])  # (b*z , 2)

        # slice_output = torch.split(classification_out, z_size)  # list of (b, z, 2)
        # slice_output = torch.stack(slice_output)
        # slice_output = torch.swapaxes(slice_output, 1 ,-1)  ## torch.Size([b, 2, z])

        # scan_output = logsumexp_attention(slice_output)  # (b,2)

        outputs = {}
        # outputs["slice_label"] = classification_out
        # outputs["scan_label"] = scan_output

        decoder_output = self.backbone.decoder(*features)
        seg_output = self.backbone.segmentation_head(decoder_output)

        seg_output = self.backbone.segmentation_head(decoder_output)[:,0,:,:]
        seg_output = torch.stack([-seg_output,seg_output],dim=1)

        outputs["mask"] = torch.swapaxes(torch.stack(torch.split(seg_output, z_size)), 1 ,2)

        return outputs["mask"]
    
def _lse_max(a, r=20, dim=0):
    log_n = torch.log(torch.tensor(a.shape[dim]).double())
    return 1 / r * (torch.logsumexp(a * r, dim=dim) - log_n)


def logsumexp_attention(a, beta=20):
    batchsize, ftrs, *shp = a.shape
    assert ftrs == 2
    a = a - a.mean(dim=1).unsqueeze(1)
    a_flat = a.view(batchsize, ftrs, -1)
    s_pos = _lse_max(a_flat[:, 1, :], r=beta, dim=1)
    s_neg = -s_pos
    s = torch.stack([s_neg, s_pos], dim=1)
    return s


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