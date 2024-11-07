from torchmetrics import Metric
import torch
import numpy as np


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [
            channel
            for channel in range(xs[0].shape[1])
            if channel not in ignore_channels
        ]
        xs = [
            torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device))
            for x in xs
        ]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def ioufunc(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union

def ioufunc_batch(pr, gt, eps=1e-7, threshold=None, ignore_channels=None, reduction='none'):
    """Calculate Intersection over Union between ground truth and prediction for batch of images
        IOU is to be calculated for each image (last 2 dimensions).
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
        reduction: 'none' for per image IOU, 'mean' for mean IOU over batch (and channels), 'sum' for sum IOU over batch (and channels)
        
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # intersection = torch.sum(gt * pr)
    # union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    # return (intersection + eps) / union
    
    intersection = torch.sum(gt * pr, dim=(1, 2)) ## B x Channels
    union = torch.sum(gt, dim=(1, 2)) + torch.sum(pr, dim=(1, 2)) - intersection + eps ## B x Channels
    iou = (intersection + eps) / union ## B x Channels
    if reduction == 'mean':
        return iou.mean()
    elif reduction == 'sum':
        return iou.sum()
    else:
        return iou


def normalize_tensor_images(tensor):
    """Normalize each image in the tensor to range [0, 1] based on its max value"""
    # Compute min and max values for each image
    min_vals = tensor.amin(dim=(1, 2), keepdim=True)
    max_vals = tensor.amax(dim=(1, 2), keepdim=True)

    # Avoid division by zero
    denominator = torch.clamp(max_vals - min_vals, min=1e-8)

    # Normalize
    normalized_tensor = (tensor - min_vals) / denominator

    return normalized_tensor


def normalize_image(img):
    """Normalize image to range [0, 1]"""
    min_val = np.min(img)
    max_val = np.max(img) 
    if max_val > min_val:
        return (img - min_val) / (max_val - min_val)
    return img  # Return original image if it's constant


class qIOU(Metric):
    higher_is_better = True

    def __init__(
        self,
        num_classes: int = 2,
        threshold=0.5,
        dist_sync_on_step=False,
        compute_on_step=False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("iou_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("iou_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Updates the IOU list by calculating the Intersection Over Union (IOU)
        for instances in the target and prediction tensors where the target
        instance is not entirely composed of zeros.

        Example:
        >>> import torch
        >>> target = torch.tensor([
        ...     [[0, 0], [0, 0]],
        ...     [[0, 1], [1, 0]],
        ...     [[1, 1], [1, 1]]
        ... ])
        >>> target > 0
        tensor([[[False, False],
                [False, False]],
                [[False,  True],
                [ True, False]],
                [[ True,  True],
                [ True,  True]]])
        >>> pos_inds = (target > 0).nonzero(as_tuple=True)[0].tolist()
        >>> #this will initially
        tensor([[1, 0, 1],
        [1, 1, 0],
        [2, 0, 0],
        [2, 0, 1],
        [2, 1, 0],
        [2, 1, 1]])
        The first row [1, 0, 1] indicates the non-zero element at the 2nd instance (index 1), row 1, column 2.
        then we return the 0th index which indicated which element contains 1 anywhere in the mask
        >>> pos_inds
        [1, 1, 2, 2, 2, 2]
        >>> index = torch.Tensor(list(set(pos_inds))).long().to(target.device)
        >>> index
        tensor([1, 2])
        >>> filtered_target = torch.index_select(target, 0, index)
        >>> filtered_target
        tensor([[[0, 1],
                [1, 0]],
                [[1, 1],
                [1, 1]]])
        >>> # ...continue with preds and IOU computation
        """
        assert preds.shape == target.shape  # this is after softmax is applied to preds
        pos_inds = (target > 0).nonzero(as_tuple=True)[0].tolist()
        index = torch.Tensor(list(set(pos_inds))).long().to(target.device)
        target = torch.index_select(target, 0, index)
        preds = torch.index_select(preds, 0, index)
        # print([(x.min(), x.max()) for x in preds])
        # preds = normalize_tensor_images(preds)
        # print([(x.min(), x.max()) for x in preds])
        if len(target) > 0:
            instance_iou = ioufunc(preds, target, self.threshold)
            self.iou_sum += instance_iou
            self.iou_count += 1
        else:
            self.iou_sum += torch.tensor(0).to(target.device)

    def compute(self):
        return self.iou_sum / ((self.iou_count) + 0.0000001)

class qIOU_corrected(Metric):
    higher_is_better = True

    def __init__(
        self,
        num_classes: int = 2,
        threshold=0.5,
        dist_sync_on_step=False,
        compute_on_step=False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("iou_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("iou_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Updates the IOU list by calculating the Intersection Over Union (IOU)
        for instances in the target and prediction tensors where the target
        instance is not entirely composed of zeros.

        Example:
        >>> import torch
        >>> target = torch.tensor([
        ...     [[0, 0], [0, 0]],
        ...     [[0, 1], [1, 0]],
        ...     [[1, 1], [1, 1]]
        ... ])
        >>> target > 0
        tensor([[[False, False],
                [False, False]],
                [[False,  True],
                [ True, False]],
                [[ True,  True],
                [ True,  True]]])
        >>> pos_inds = (target > 0).nonzero(as_tuple=True)[0].tolist()
        >>> #this will initially
        tensor([[1, 0, 1],
        [1, 1, 0],
        [2, 0, 0],
        [2, 0, 1],
        [2, 1, 0],
        [2, 1, 1]])
        The first row [1, 0, 1] indicates the non-zero element at the 2nd instance (index 1), row 1, column 2.
        then we return the 0th index which indicated which element contains 1 anywhere in the mask
        >>> pos_inds
        [1, 1, 2, 2, 2, 2]
        >>> index = torch.Tensor(list(set(pos_inds))).long().to(target.device)
        >>> index
        tensor([1, 2])
        >>> filtered_target = torch.index_select(target, 0, index)
        >>> filtered_target
        tensor([[[0, 1],
                [1, 0]],
                [[1, 1],
                [1, 1]]])
        >>> # ...continue with preds and IOU computation
        """
        assert preds.shape == target.shape  # this is after softmax is applied to preds
        pos_inds = (target > 0).nonzero(as_tuple=True)[0].tolist()
        index = torch.Tensor(list(set(pos_inds))).long().to(target.device)
        target = torch.index_select(target, 0, index)
        preds = torch.index_select(preds, 0, index)
        # print([(x.min(), x.max()) for x in preds])
        # preds = normalize_tensor_images(preds)
        # print([(x.min(), x.max()) for x in preds])
        if len(target) > 0:
            instance_iou = ioufunc_batch(preds, target, self.threshold, reduction='none')
            count = instance_iou.size(0)
            self.iou_sum += instance_iou.sum()
            self.iou_count += count
        else:
            self.iou_sum += torch.tensor(0).to(target.device)

    def compute(self):
        return self.iou_sum / ((self.iou_count) + 0.0000001)