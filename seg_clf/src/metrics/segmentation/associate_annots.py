from typing import List, Tuple

import numpy as np
from qct_utils.ct_schema import Ctscan, CtscanAnnot, PredBoxType
from qct_utils.sitk_ops import sitk_rle2mask


def associate_annots(
    gt_ctscan: Ctscan, pred_ctscan: Ctscan, iou_thr: float = 0.3
) -> Tuple[Ctscan, Ctscan]:
    """Associate Box annotations in gt_ctscan and pred_ctscan.
    Where each annotation box in
    1. 'gt_ctscan' will be updated to TP or FN
    2. ''pred_ctscan' will be updated to TP or FP

    IOU is calculated for all pairs (gt_boxes, pred_boxes). Less IOUs are discarded.
    Others are sorted by their confidence scores and matched.
    1. Unmatched in 'gt_boxes' to FN
    2. Unmatched in 'pred_boxes' to FP
    3. Matched to TP

    Args:
        gt_ctscan (Ctscan): Ground truth ctscan
        pred_ctscan (Ctscan): Prediction ctscan
        iou_thr (float, optional):  IOU threshold for a prediction box to be positive. Defaults to 0.3.

    Returns:
        Tuple[Ctscan, Ctscan]: returns tuple of (gt_ctscan, pred_ctscan) with their PredBoxes updated.
    """

    gt_annots = gt_ctscan.Annot
    if gt_annots is None:
        gt_annots = []
    pred_annots = pred_ctscan.Annot
    if pred_annots is None:
        pred_annots = []
    all_matches = np.empty((0, 3))
    if (len(gt_annots) > 0) and (len(pred_annots) > 0):
        all_ious = cal_box_iou_matrix(gt_annots, pred_annots)
        want_idx = np.where(all_ious > iou_thr)
        all_matches = []
        for i in range(want_idx[0].shape[0]):
            all_matches.append(
                [
                    want_idx[0][i],
                    want_idx[1][i],
                    all_ious[want_idx[0][i], want_idx[1][i]],
                ]
            )
        # [N, 3]: i, j, score. [gt_index, pred_index, iou_score]
        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

    for num, gt_ctscanannot in enumerate(gt_annots):
        TP_FP_FN = "FN"
        if len(all_matches) > 0:
            if num in all_matches[:, 0].tolist():
                TP_FP_FN = "TP"
        gt_ctscanannot.pred_annot = PredBoxType(
            TP_FP_FN=TP_FP_FN, gt_Nodule_name=gt_ctscanannot.Nodule_name
        )

    for num, pred_ctscaannot in enumerate(pred_annots):
        iou_score, TP_FP_FN = 0.0, "FP"
        iou_score = 0.0
        gt_Nodule_name = ""
        if len(all_matches) > 0:
            if num in np.int64(all_matches[:, 1]):
                index = int(np.where(np.int64(all_matches[:, 1]) == num)[0])
                gt_Nodule_name = gt_annots[int(all_matches[index][0])].Nodule_name
                iou_score = all_matches[index][2]
                TP_FP_FN = "TP"
        pred_ctscaannot.pred_annot = PredBoxType(
            TP_FP_FN=TP_FP_FN, gt_Nodule_name=gt_Nodule_name, IOU=iou_score
        )

    return gt_ctscan, pred_ctscan


def mask_iou_matrix_ctscan_annots(
    gt_ctscan_annots: List[CtscanAnnot], pred_ctscan_annots: List[CtscanAnnot]
):
    gt_masks = np.asarray(
        [np.int64(sitk_rle2mask(a.annot.rle, a.annot.size)) for a in gt_ctscan_annots]
    )
    pred_masks = np.asarray(
        [np.int64(sitk_rle2mask(a.annot.rle, a.annot.size)) for a in pred_ctscan_annots]
    )

    def get_full_mask(masks):
        mask = np.zeros_like(masks[0], dtype=np.uint8)
        for i, m in enumerate(masks):
            mask[m > 0] = i + 1
        return mask

    gt_mask = get_full_mask(gt_masks)
    pred_mask = get_full_mask(pred_masks)

    return mask_iou_matrix_numpy(gt_mask, pred_mask)


def mask_iou_matrix_numpy(labels: np.ndarray, y_pred: np.ndarray, SMOOTH=1e-6):
    # Compute number of objects
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / (union + SMOOTH)
    return iou


def iou_dice_precision_numpy(outputs: np.ndarray, labels: np.ndarray, SMOOTH=1e-6):
    """[Ignores batch dimension.] Assumes single mask."""
    outputs = outputs.astype(bool)
    labels = labels.astype(bool)
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    dice = (2 * intersection) / (intersection + union + SMOOTH)
    pixel_precision = (intersection) / (outputs.sum() + SMOOTH)
    return iou, dice, pixel_precision


def py_box_overlap(boxes1, boxes2):
    overlap = np.zeros((len(boxes1), len(boxes2)))

    z1, y1, x1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2]
    d1, h1, w1 = boxes1[:, 3], boxes1[:, 4], boxes1[:, 5]
    areas1 = d1 * h1 * w1

    z2, y2, x2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2]
    d2, h2, w2 = boxes2[:, 3], boxes2[:, 4], boxes2[:, 5]
    areas2 = d2 * h2 * w2

    for i in range(len(boxes1)):
        xx0 = np.maximum(x1[i] - w1[i] / 2.0, x2 - w2 / 2.0)
        yy0 = np.maximum(y1[i] - h1[i] / 2.0, y2 - h2 / 2.0)
        zz0 = np.maximum(z1[i] - d1[i] / 2.0, z2 - d2 / 2.0)
        xx1 = np.minimum(x1[i] + w1[i] / 2.0, x2 + w2 / 2.0)
        yy1 = np.minimum(y1[i] + h1[i] / 2.0, y2 + h2 / 2.0)
        zz1 = np.minimum(z1[i] + d1[i] / 2.0, z2 + d2 / 2.0)

        inter_w = np.maximum(0.0, xx1 - xx0)
        inter_h = np.maximum(0.0, yy1 - yy0)
        inter_d = np.maximum(0.0, zz1 - zz0)
        intersect = inter_w * inter_h * inter_d
        overlap[i] = intersect / (areas1[i] + areas2 - intersect)

    return overlap


def cal_box_iou_matrix(gt: CtscanAnnot, pred: CtscanAnnot):
    gt_bbox = np.asarray(
        [
            [
                a.annot.bbox.z_center,
                a.annot.bbox.y_center,
                a.annot.bbox.x_center,
                a.annot.bbox.d,
                a.annot.bbox.h,
                a.annot.bbox.w,
            ]
            for a in gt
        ]
    )
    pred_bbox = np.asarray(
        [
            [
                a.annot.conf,
                a.annot.bbox.z_center,
                a.annot.bbox.y_center,
                a.annot.bbox.x_center,
                a.annot.bbox.d,
                a.annot.bbox.h,
                a.annot.bbox.w,
            ]
            for a in pred
        ]
    )
    iou = py_box_overlap(gt_bbox, pred_bbox[:, 1:])
    return iou
