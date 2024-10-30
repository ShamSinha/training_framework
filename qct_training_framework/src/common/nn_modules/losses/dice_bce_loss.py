from typing import List

import torch
import torch.nn as nn
from loguru import logger
from monai.networks import one_hot


def dice_loss(input: torch.Tensor, target: torch.Tensor):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def bce_dice(
    input: torch.Tensor,
    target: torch.Tensor,
    loss_weights: List[float] = [1, 1],
    to_onehot_y: bool = False,
):
    if to_onehot_y:
        n_pred_ch = input.shape[1]
        if n_pred_ch == 1:
            logger.warning("single channel prediction, `to_onehot_y=True` ignored.")
        else:
            target = one_hot(target, num_classes=n_pred_ch)
    loss1 = loss_weights[0] * nn.BCEWithLogitsLoss()(input, target)
    loss2 = loss_weights[1] * dice_loss(input, target)
    return (loss1 + loss2) / sum(loss_weights)
