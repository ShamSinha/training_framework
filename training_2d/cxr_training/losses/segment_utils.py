import torch.nn as nn
import torch
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1)[:, 1, :, :].contiguous()

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, ce_weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        if ce_weight is not None:
            self.CE_loss = nn.CrossEntropyLoss(
                torch.FloatTensor([1, ce_weight]), reduction="mean"
            )
        else:
            self.CE_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, inputs, target_tensor, dice_threshold=0.2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        BCE = self.CE_loss(inputs, target_tensor)
        inputs = F.softmax(inputs, dim=1)[:, 1, :, :].contiguous()

        # flatten label and prediction tensors
        targets = target_tensor * (target_tensor >= 0)
        inputs = inputs * (target_tensor >= 0)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )

        Dice_BCE = BCE + dice_loss * dice_threshold

        return {'bce_loss': BCE, 'dice_loss': dice_loss, 'dice_bce':Dice_BCE}


class SquaredDiceBCELoss(nn.Module):
    def __init__(self, ce_weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        if ce_weight is not None:
            self.CE_loss = nn.CrossEntropyLoss(
                torch.FloatTensor([1, ce_weight]), reduction="mean"
            )
        else:
            self.CE_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, inputs, target_tensor, dice_threshold=0.2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        BCE = self.CE_loss(inputs, target_tensor)
        inputs = F.softmax(inputs, dim=1)[:, 1, :, :].contiguous()

        # flatten label and prediction tensors
        targets = target_tensor * (target_tensor >= 0)
        inputs = inputs * (target_tensor >= 0)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            (inputs**2).sum() + (targets**2).sum() + smooth
        )

        Dice_BCE = BCE + dice_loss * dice_threshold

        return {'bce_loss': BCE, 'dice_loss': dice_loss, 'dice_bce':Dice_BCE}


class DiceBCELossPix(nn.Module):
    def __init__(self, ce_weight=None, reduction="none"):
        super(DiceBCELossPix, self).__init__()
        if ce_weight is not None:
            self.CE_loss = nn.CrossEntropyLoss(
                torch.FloatTensor([1, ce_weight]), reduction=reduction
            )
        else:
            self.CE_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, target_tensor, dice_threshold=0.2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        BCE = self.CE_loss(inputs, target_tensor)
        inputs = F.softmax(inputs, dim=1)[:, 1, :, :].contiguous()
        smooth = smooth / target_tensor.size()[0]

        # flatten label and prediction tensors
        targets = target_tensor * (target_tensor >= 0)
        inputs = inputs * (target_tensor >= 0)
        intersection = (inputs * targets).sum([1, 2])
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum([1, 2]) + targets.sum([1, 2]) + smooth
        )

        Dice_BCE = BCE.mean([1, 2]) + dice_loss * dice_threshold

        return {'bce_loss': BCE.mean([1, 2]), 'dice_loss': dice_loss, 'dice_bce': Dice_BCE}
    

class DiceBCELossPix_seg(nn.Module):
    def __init__(self, ce_weight=None, reduction="mean"):
        super(DiceBCELossPix, self).__init__()
        if ce_weight is not None:
            self.CE_loss = nn.CrossEntropyLoss(
                torch.FloatTensor([1, ce_weight]), reduction=reduction
            )
        else:
            self.CE_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, target_tensor, dice_threshold=0.2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        BCE = self.CE_loss(inputs, target_tensor)
        inputs = F.softmax(inputs, dim=1)[:, 1, :, :].contiguous()
        smooth = smooth / target_tensor.size()[0]

        # flatten label and prediction tensors
        targets = target_tensor * (target_tensor >= 0)
        inputs = inputs * (target_tensor >= 0)
        intersection = (inputs * targets).sum([1, 2])
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum([1, 2]) + targets.sum([1, 2]) + smooth
        )

        Dice_BCE = BCE.mean() + dice_loss.mean() * dice_threshold

        # return Dice_BCE
        return {'bce_loss': BCE.mean(), 'dice_loss': dice_loss.mean(), 'dice_bce':Dice_BCE}