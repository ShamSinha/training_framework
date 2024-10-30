from matplotlib.pyplot import grid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np
from functools import partial

from timm.models.vision_transformer import Block
from unittest.mock import patch
from timm.layers.helpers import to_2tuple, to_3tuple
from loguru import logger
from transformers import PvtModel, PvtConfig


class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        in_chan_last=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            self.grid_size.append(im_size // pa_size)
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.in_chans = in_chans
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten
        self.in_chan_last = in_chan_last

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, L, S = x.shape
        assert (
            S == np.prod(self.img_size) * self.in_chans
        ), f"Input image total size {S} doesn't match model configuration"
        if self.in_chan_last:
            x = x.reshape(B * L, *self.img_size, self.in_chans).permute(
                0, 3, 1, 2
            )  # When patchification follows HWC
        else:
            x = x.reshape(B * L, self.in_chans, *self.img_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        norm_layer=None,
        flatten=True,
        in_chan_last=True,
    ):
        super().__init__()
        img_size = img_size
        patch_size = patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            self.grid_size.append(im_size // pa_size)
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.in_chans = in_chans
        self.num_patches = np.prod(self.grid_size)

        self.flatten = flatten
        self.in_chan_last = in_chan_last

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        B, L, S = x.shape
        assert (
            S == np.prod(self.img_size) * self.in_chans
        ), f"Input image total size {S} doesn't match model configuration"
        if self.in_chan_last:
            x = x.reshape(B * L, *self.img_size, self.in_chans).permute(
                0, 4, 1, 2, 3
            )  # When patchification follows HWDC
        else:
            x = x.reshape(B * L, self.in_chans, *self.img_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x


def build_2d_sincos_position_embedding(
    grid_size, embed_dim, num_tokens=1, temperature=10000.0
):
    """
    TODO: the code can work when grid size is isotropic (H==W), but it is not logically right especially when data is non-isotropic(H!=W).
    """
    h, w = grid_size, grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert (
        embed_dim % 4 == 0
    ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
    )[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


class MAEViTEncoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage
    Modified from timm implementation
    """

    def __init__(
        self,
        embed_layer,
        patch_size,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        use_pe=True,
        return_patchembed=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1  # don't consider distillation here
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.use_pe = use_pe
        self.return_patchembed = return_patchembed

        self.patch_embed = embed_layer
        # assert self.patch_embed.num_patches == 1, \
        #         "Current embed layer should output 1 token because the patch length is reshaped to batch dimension"

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.cls_pe = nn.Parameter(torch.zeros([1, 1, embed_dim], dtype=torch.float32))
        # self.cls_pe.requires_grad = False
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # init patch embed parameters
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.cls_token, std=0.02)
        # trunc_normal_(self.cls_token, std=.02, a=-.02, b=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward_features(self, x, pos_embed=None):
        return_patchembed = self.return_patchembed

        embed_dim = self.embed_dim

        # logger.debug(x.shape)
        B, L, _ = x.shape

        x = self.patch_embed(x)  # [B*L, embed_dim]
        x = x.reshape(B, L, embed_dim)

        # logger.debug(x.shape)

        if return_patchembed:
            patchembed = x
        cls_token = self.cls_token.expand(B, -1, -1)

        # logger.debug(cls_token.shape)

        x = torch.cat((cls_token, x), dim=1)

        # logger.debug(x.shape)

        if self.use_pe:
            if x.size(1) != pos_embed.size(1):
                assert x.size(1) == pos_embed.size(1) + 1, "Unmatched x and pe shapes"
                cls_pe = torch.zeros([B, 1, embed_dim], dtype=torch.float32).to(
                    x.device
                )
                pos_embed = torch.cat([cls_pe, pos_embed], dim=1)
            x = self.pos_drop(x + pos_embed)
            # logger.debug(x.shape)

        for blk in self.blocks:
            x = blk(x)

        # logger.debug(x.shape)

        x = self.norm(x)
        # logger.debug(x.shape)
        if return_patchembed:
            return x, patchembed
        else:
            return x

    def forward(self, x, pos_embed=None):
        if self.return_patchembed:
            x, patch_embed = self.forward_features(x, pos_embed)
        else:
            x = self.forward_features(x, pos_embed)
        x = self.head(x)
        if self.return_patchembed:
            return x, patch_embed
        else:
            return x


class MAEViTDecoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage
    Modified from timm implementation
    """

    def __init__(
        self,
        patch_size,
        num_classes=768,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == 3 * patch_size ** 2
        self.embed_dim = embed_dim
        self.num_tokens = 1  # don't consider distillation here
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def mae_encoder_small_patch16_224(**kwargs):
    model = MAEViTEncoder(embed_dim=384, num_heads=6, **kwargs)
    return model


def mae_decoder_small_patch16_224(**kwargs):
    model = MAEViTDecoder(embed_dim=128, depth=4, num_heads=3, **kwargs)
    return model


def build_3d_sincos_position_embedding(
    grid_size, embed_dim, num_tokens=1, temperature=10000.0
):
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)
    assert (
        embed_dim % 6 == 0
    ), "Embed dimension must be divisible by 6 for 3D sin-cos position embedding"
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_d = torch.einsum("m,d->md", [grid_d.flatten(), omega])
    pos_emb = torch.cat(
        [
            torch.sin(out_h),
            torch.cos(out_h),
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_d),
            torch.cos(out_d),
        ],
        dim=1,
    )[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


def build_perceptron_position_embedding(grid_size, embed_dim, num_tokens=1):
    pos_emb = torch.rand([1, np.prod(grid_size), embed_dim])
    nn.init.normal_(pos_emb, std=0.02)

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    return pos_embed


def patchify_image(x, patch_size):
    """
    ATTENTION!!!!!!!
    Different from 2D version patchification: The final axis follows the order of [ph, pw, pd, c] instead of [c, ph, pw, pd]
    """
    # patchify input, [B,C,H,W,D] --> [B,C,gh,ph,gw,pw,gd,pd] --> [B,gh*gw*gd,ph*pw*pd*C]
    B, C, H, W, D = x.shape
    grid_size = (H // patch_size[0], W // patch_size[1], D // patch_size[2])

    x = x.reshape(
        B,
        C,
        grid_size[0],
        patch_size[0],
        grid_size[1],
        patch_size[1],
        grid_size[2],
        patch_size[2],
    )  # [B,C,gh,ph,gw,pw,gd,pd]
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
        B, np.prod(grid_size), np.prod(patch_size) * C
    )  # [B,gh*gw*gd,ph*pw*pd*C]

    return x


def batched_shuffle_indices(batch_size, length, device):
    """
    Generate random permutations of specified length for batch_size times
    Motivated by https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
    """
    rand = torch.rand(batch_size, length).to(device)
    batch_perm = rand.argsort(dim=1)
    return batch_perm


class MAE3D(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        mask_ratio=0.75,
        pos_embed_type="sincos",
        input_size=[48, 48, 16],
        patch_size=[6, 6, 4],
        in_chans=1,
        encoder_embed_dim=768,
        encoder_depth=6,
        encoder_num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=12,
        encoder_drop_rate=0.0,
        attn_drop_rate = 0.0
    ):
        super().__init__()
        input_size = tuple(input_size)
        patch_size = tuple(patch_size)
        self.input_size = input_size
        self.patch_size = patch_size

        out_chans = in_chans * np.prod(self.patch_size)
        self.out_chans = out_chans

        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, "input size and patch size are not proper"
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size

        # build positional encoding for encoder and decoder
        if pos_embed_type == "sincos":
            with torch.no_grad():
                self.encoder_pos_embed = build_3d_sincos_position_embedding(
                    grid_size, encoder_embed_dim, num_tokens=0
                )
                self.decoder_pos_embed = build_3d_sincos_position_embedding(
                    grid_size, decoder_embed_dim, num_tokens=0
                )
        elif pos_embed_type == "perceptron":
            self.encoder_pos_embed = build_perceptron_position_embedding(
                grid_size, encoder_embed_dim, num_tokens=0
            )
            with torch.no_grad():
                self.decoder_pos_embed = build_3d_sincos_position_embedding(
                    grid_size, decoder_embed_dim, num_tokens=0
                )

        # build encoder and decoder
        embed_layer = PatchEmbed3D(
            img_size=patch_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
        )
        self.encoder = MAEViTEncoder(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            embed_layer=embed_layer,
            drop_rate=encoder_drop_rate,
            attn_drop_rate=attn_drop_rate
        )
        self.decoder = MAEViTDecoder(
            patch_size=patch_size,
            num_classes=out_chans,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
        )

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=True
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.patch_norm = nn.LayerNorm(
            normalized_shape=(out_chans,), eps=1e-6, elementwise_affine=False
        )

        self.criterion = nn.MSELoss()

        # initialize encoder_to_decoder and mask token
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        nn.init.normal_(self.mask_token, std=0.02)

        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim

    def unpatchify(self, x):
        batch_size = x.shape[0]
        H, W, D = self.input_size
        Ph, Pw, Pd = self.patch_size
        num_patches_h = H // Ph
        num_patches_w = W // Pw
        num_patches_d = D // Pd
        unshuffled_all_x = x.reshape(
            batch_size, num_patches_h, num_patches_w, num_patches_d, self.in_chans, Ph, Pw, Pd
        )
        unshuffled_all_x = unshuffled_all_x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            batch_size, self.in_chans, H, W, D
        )

        final_x = unshuffled_all_x.squeeze()
        final_x = final_x.permute(0,3,1,2)
        # final_x = unshuffled_all_x.permute(0,1, 4, 2, 3)
        return final_x

    def get_classification_embed(self, x):
        batch_size = x.size(0)
        in_chans = x.size(1)
        assert in_chans == self.in_chans

        x = patchify_image(x, self.patch_size)  # [B,gh*gw*gd,ph*pw*pd*C]

        sel_x = x

        # Use the original position embeddings for all patches
        sel_encoder_pos_embed = self.encoder_pos_embed.expand(batch_size, -1, -1)

        # logger.debug(sel_encoder_pos_embed.shape)

        # Forward through the encoder
        sel_x = self.encoder(sel_x, sel_encoder_pos_embed)

        cls_embedding = sel_x[:, 0, :]

        return cls_embedding

    def forward(self, x, apply_mask=True, return_image=False):
        batch_size = x.size(0)
        in_chans = x.size(1)
        assert in_chans == self.in_chans

        # logger.debug(x.shape)

        out_chans = self.out_chans
        x = patchify_image(x, self.patch_size)  # [B,gh*gw*gd,ph*pw*pd*C]

        # logger.debug(x.shape)

        if apply_mask:
            # compute length for selected and masked
            length = np.prod(self.grid_size)

            # logger.debug(length)

            sel_length = int(length * (1 - self.mask_ratio))
            msk_length = length - sel_length

            # logger.debug(msk_length)

            # generate batched shuffle indices
            shuffle_indices = batched_shuffle_indices(
                batch_size, length, device=x.device
            )
            unshuffle_indices = shuffle_indices.argsort(dim=1)

            # select and mask the input patches
            shuffled_x = x.gather(
                dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, out_chans)
            )
            # logger.debug(shuffled_x.shape)

            sel_x = shuffled_x[:, :sel_length, :]
            msk_x = shuffled_x[:, -msk_length:, :]

            # logger.debug(sel_x.shape)
            # logger.debug(msk_x.shape)

            # select and mask the indices
            # shuffle_indices = F.pad(shuffle_indices + 1, pad=(1, 0), mode='constant', value=0)
            sel_indices = shuffle_indices[:, :sel_length]
            # msk_indices = shuffle_indices[:, -msk_length:]

            # select the position embedings accordingly
            sel_encoder_pos_embed = self.encoder_pos_embed.expand(
                batch_size, -1, -1
            ).gather(
                dim=1,
                index=sel_indices[:, :, None].expand(-1, -1, self.encoder_embed_dim),
            )

            # logger.debug(sel_encoder_pos_embed.shape)

            # forward encoder & proj to decoder dimension
            sel_x = self.encoder(sel_x, sel_encoder_pos_embed)

            cls_embedding = sel_x[:, 0, :]
            # logger.debug(sel_x.shape)

            sel_x = self.encoder_to_decoder(sel_x)

            # logger.debug(sel_x.shape)

            # combine the selected tokens and mask tokens in the shuffled order
            all_x = torch.cat(
                [sel_x, self.mask_token.expand(batch_size, msk_length, -1)], dim=1
            )

            # logger.debug(all_x.shape)

            # shuffle all the decoder positional encoding
            shuffled_decoder_pos_embed = self.decoder_pos_embed.expand(
                batch_size, -1, -1
            ).gather(
                dim=1,
                index=shuffle_indices[:, :, None].expand(
                    -1, -1, self.decoder_embed_dim
                ),
            )

            # logger.debug(shuffled_decoder_pos_embed.shape)

            # add the shuffled positional embedings to encoder output tokens
            all_x[:, 1:, :] += shuffled_decoder_pos_embed
            # all_x = all_x + shuffled_decoder_pos_embed

            # logger.debug(all_x.shape)

            # forward decoder
            all_x = self.decoder(all_x)

            # logger.debug(all_x.shape)

            # logger.debug(all_x[:, -msk_length:, :].shape)

            # loss
            loss = self.criterion(
                input=all_x[:, -msk_length:, :], target=self.patch_norm(msk_x.detach())
            )

        else:
            sel_x = x

            # Use the original position embeddings for all patches
            sel_encoder_pos_embed = self.encoder_pos_embed.expand(batch_size, -1, -1)

            # logger.debug(sel_encoder_pos_embed.shape)

            # Forward through the encoder
            sel_x = self.encoder(sel_x, sel_encoder_pos_embed)

            cls_embedding = sel_x[:, 0, :]

            # logger.debug(sel_x.shape)

            # Project encoder outputs to decoder dimension
            all_x = self.encoder_to_decoder(sel_x)

            # logger.debug(all_x.shape)

            # Decoder positional embeddings
            decoder_pos_embed = self.decoder_pos_embed.expand(batch_size, -1, -1)

            # logger.debug(decoder_pos_embed.shape)

            # Add positional embeddings to the encoder output
            all_x[:, 1:, :] += decoder_pos_embed

            # Forward through the decoder
            all_x = self.decoder(all_x)

            # logger.debug(all_x.shape)

            # Logic for returning images when masking is NOT applied
            # We reconstruct the image from the output of the decoder
            recon = all_x[:, 1:, :]
            # logger.debug(recon.shape)

            # Optionally, scale the reconstructed patches back to the original scale
            recon = recon * (
                x.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6
            ) + x.mean(dim=-1, keepdim=True)

            # Convert the patches back to the image format
            recon_image = self.unpatchify(recon.detach())
            input_image = self.unpatchify(x.detach())

            loss = self.criterion(input=recon_image, target=input_image)

        if return_image:
            if apply_mask:
                # unshuffled all the tokens
                masked_x = torch.cat(
                    [
                        shuffled_x[:, :sel_length, :],
                        0.0
                        * torch.ones(batch_size, msk_length, out_chans).to(x.device),
                    ],
                    dim=1,
                ).gather(
                    dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans)
                )
                recon = all_x[:, 1:, :].gather(
                    dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans)
                )
                recon = recon * (
                    x.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6
                ) + x.mean(dim=-1, keepdim=True)

                recon_image = self.unpatchify(recon.detach())
                masked_input = self.unpatchify(masked_x.detach())

                return loss, cls_embedding.detach(), recon_image, masked_input
            else:
                return loss, cls_embedding.detach(), recon_image
        else:
            return loss
