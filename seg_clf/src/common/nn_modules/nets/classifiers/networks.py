import monai.networks.nets as monainets  # noqa
import torch.nn as nn

from .layers import Encoder, GroupViews, MVTail

__all__ = ["MVModelClassifier", "monainets"]


class MVModelClassifier(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_channels: int,
        drop_rate=0.1,
        act_layer: str = "nn.ReLU",
        norm="group",
        total_steps: int = int(10e3),
        drop_type: str = "dropout",
    ):
        super().__init__()
        self.in_chs = in_chs
        self.out_channels = out_channels
        self.drop_rate = drop_rate
        self.act_layer = eval(act_layer)  # nosec
        self.norm = norm
        self.total_steps = total_steps
        self.drop_type = drop_type

        self._setup_model()

    def _setup_model(
        self,
    ):
        # axial, coronal, sagittal, min mip, max mip encoder
        self.encoder = Encoder(
            in_chs=self.in_chs,
            act_layer=self.act_layer,
            drop_prob=self.drop_rate,
            drop_block_size=3,
            total_steps=self.total_steps,
            norm=self.norm,
            drop_type=self.drop_type,
        )
        self.gv_axial = GroupViews(
            32,
            3,
            act_layer=self.act_layer,
            drop_prob=self.drop_rate,
            drop_block_size=3,
            norm=self.norm,
            total_steps=self.total_steps,
            drop_type=self.drop_type,
        )
        self.gv_ax_c = GroupViews(
            32,
            2,
            act_layer=self.act_layer,
            drop_prob=self.drop_rate,
            drop_block_size=3,
            norm=self.norm,
            total_steps=self.total_steps,
            drop_type=self.drop_type,
        )
        self.gv_ax_seg = GroupViews(
            32,
            2,
            act_layer=self.act_layer,
            drop_prob=self.drop_rate,
            drop_block_size=3,
            norm=self.norm,
            total_steps=self.total_steps,
            drop_type=self.drop_type,
        )
        self.gv = GroupViews(
            64,
            3,
            act_layer=self.act_layer,
            drop_prob=self.drop_rate,
            drop_block_size=3,
            norm=self.norm,
            total_steps=self.total_steps,
            drop_type=self.drop_type,
        )
        self.mv_tail = MVTail(
            128,
            act_layer=self.act_layer,
            drop_rate=self.drop_rate,
            out_channels=self.out_channels,
            drop_block_size=3,
            norm=self.norm,
            total_steps=self.total_steps,
            drop_type=self.drop_type,
        )

    def forward(self, x):
        axial, coronal, sagittal, min_mip, max_mip = x
        axial_op = self.encoder(axial)
        coronal_op = self.encoder(coronal)
        sagittal_op = self.encoder(sagittal)
        min_mip_op = self.encoder(min_mip)
        max_mip_op = self.encoder(max_mip)

        gv_1 = self.gv_axial([axial_op, min_mip_op, max_mip_op])
        gv_2 = self.gv_ax_c([axial_op, coronal_op])
        gv_3 = self.gv_ax_seg([axial_op, sagittal_op])
        gv_op = self.gv([gv_1, gv_2, gv_3])

        return self.mv_tail(gv_op)
