import torch
import torch.nn as nn
from cxr_training.nnmodule.models.CBAM_head import CBAM


class MultiHead(nn.Module):
    def __init__(self, model_architecture, args):
        """generates a model with multiple classification and segmentation heads (side heads if in recipe).
        Args:
            arch (nn.Module): Model architecture
            args (argparse): Arguments needed to create the model
        """
        super().__init__()
        self.args = args
        self.cls_tags = self.args.cls.heads
        self.seg_tags = self.args.seg.heads
        self.im_size = self.args.params.im_size
        self.main_arch = model_architecture
        self.recipe = args.trainer.recipe
        cls_head_type = getattr(self.args.model, "cls_head_type", "normal")
        self.cls_feature_head = ClsHead_spinal if  cls_head_type == "spinal" else ClsHead
        self.seg_feature_head = SegHead
        self.age_feature_head = AgeHead

        if "cls" not in self.recipe:
            self.cls_tags = []
        elif "seg" not in self.recipe:
            self.seg_tags = []

        enc_ch, dec_ch = self.get_channels(args)

        if "cls" in self.recipe:
            self.cls_heads = nn.ModuleList(
                [self.cls_feature_head(enc_ch) for cl in self.cls_tags]
            )

        if "seg" in self.recipe:
            self.seg_heads = nn.ModuleList(
                [
                    self.seg_feature_head(in_channels=dec_ch, im_size=self.im_size)
                    for cl in self.seg_tags
                ]
            )

        if "age" in self.recipe:
            self.age_head = self.age_feature_head(enc_ch)

    def get_channels(self, args):
        """returns output channels of encoder -> to be passed to classification heads and
        output channels of decoder -> to be passed to segmentation heads
        """
        dummy_im = torch.randn(
            4, 1, self.im_size, self.im_size
        )  # batchno*no_of_channels*size*size
        with torch.no_grad():
            dummy_enc_out = self.main_arch.encoder(dummy_im)
            dummy_dec_out = self.main_arch.decoder(*dummy_enc_out)
        return dummy_enc_out[-1].shape[1], dummy_dec_out.shape[1]

    def forward(self, input_):
        """returns two dicts one each for segmentation and classification heads,
           the two dicts are updated with super heads with keys f"{tag}_main" where
           tag -> super_tags

        Args:
            input_ ([torch tensor]): input images

        Returns:
            [tuple]: returns tuple containing two dicts,(cls_dict, seg_dict)
        """
        enc_output = self.main_arch.encoder(input_)
        cls_out_dict = {}
        seg_out_dict = {}
        age_output = []
        if "cls" in self.recipe:
            cls_out = [head(enc_output[-1]) for head in self.cls_heads]
            cls_out_dict = {
                f"{self.cls_tags[i]}": cls_out[i] for i in range(len(self.cls_tags))
            }

        if "age" in self.recipe:
            age_output = self.age_head(enc_output[-1])

        if "seg" in self.recipe:
            dec_out = self.main_arch.decoder(*enc_output)
            seg_out = [head(dec_out) for head in self.seg_heads]
            seg_out_dict = {
                f"{self.seg_tags[i]}": seg_out[i] for i in range(len(self.seg_tags))
            }

        return {
            "classification_out": cls_out_dict,
            "segmentation_out": seg_out_dict,
            "age_out": age_output,
        }


class AgeHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=1):
        super(AgeHead, self).__init__()
        self.lungconv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.lungconv(x)
        x = x.view(*x.size()[:2])
        x = self.relu(self.fc(x))
        return x



half_in_size = 256
layer_width = 400

class ClsHead_spinal(nn.Module):
    def __init__(self, in_channels=512, out_channels=2):
        """gives detached classification heads, can be attached to a model.

        Args:
            in_channels (int, optional): number of input channels to the head. Defaults to 512.
            out_channels (int, optional): number of outpur channels it gives. Defaults to 2.
        """
        super(ClsHead_spinal, self).__init__()
        self.lungconv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.SELU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(p=0.35)      
        self.fc_spinal_layer1 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.SELU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width*1, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.SELU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width*2, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.SELU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width*3, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.SELU(inplace=True),)
        self.fc_out = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width*4, 2),)


    def forward(self, x):
        x = self.lungconv(x)
        x = x.view(x.size()[0], -1)
        xOrgD = self.dropout(x)
        x1 = self.fc_spinal_layer1(x)
        xOrgD = torch.cat([xOrgD, x1], dim=1)
        x2 = self.fc_spinal_layer2(xOrgD)
        xOrgD = torch.cat([xOrgD, x2], dim=1)
        x3 = self.fc_spinal_layer3(xOrgD)
        xOrgD = torch.cat([xOrgD, x3], dim=1)
        x4 = self.fc_spinal_layer4(xOrgD)
        x = torch.cat([xOrgD, x4], dim=1)
        x = self.fc_out(x)
        return x

class ClsHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=2):
        """gives detached classification heads, can be attached to a model.

        Args:
            in_channels (int, optional): number of input channels to the head. Defaults to 512.
            out_channels (int, optional): number of outpur channels it gives. Defaults to 2.
        """
        super(ClsHead, self).__init__()
        self.lungconv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.lungconv(x)
        x = x.view(*x.size()[:2])
        x = self.fc(x)
        return x


class SegHead(nn.Module):
    def __init__(self, in_channels=128, out_channels=2, im_size=960):
        """gives detached segmentation heads, can be attached to a model.

        Args:
            in_channels (int, optional): number of input channels to the head. Defaults to 128.
            out_channels (int, optional): number of outpur channels it gives. Defaults to 2.
            im_size (int, optional): input image size. Defaults to 960.
        """
        super(SegHead, self).__init__()
        self.seghead = nn.Sequential(
            CBAM(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, stride=1),
            nn.UpsamplingBilinear2d(size=(im_size, im_size)),
        )

    def forward(self, input_):
        out = self.seghead(input_)
        return out
