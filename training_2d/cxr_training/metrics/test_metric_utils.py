from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import cv2
import os
import numpy as np
from tqdm import tqdm as pbar
import glob
import pickle


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
def get_auc_score(target, prob):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

    return roc_auc_score(target, prob[:, 1])
    # used as a function to change the device of the data instead of to.device


def get_accuracy(preds, target):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    return accuracy_score(target, preds)

    # used as a function to change the device of the data instead of to.device


def get_cutoff_youdens_index(fpr, tpr, thresholds) -> float:
    """Calculate and return the threshold value using Youden's Index."""
    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]


def iou_dirs(preds_dir, gt_dirs, tag):
    ## Modifying ot for list of gt_dirs
    preds_dict = {}
    gt_dict = {}

    preds_dict = {
        os.path.basename(x).replace(".png", ""): x for x in glob.glob(f"{preds_dir}/{tag}/*")
    }

    for gt_dir in gt_dirs:
        gt_dict.update({
            os.path.basename(x).replace(".png", ""): x for x in glob.glob(f"{gt_dir}/{tag}/*")
        })

    iou_dat = {}

    thresholds = np.arange(0.1, 1, 0.05)
    print(f"The thresholds given are {thresholds}")

    for _batch_id, idx in enumerate(pbar(preds_dict)):
        if idx in gt_dict:
            pred = cv2.imread(preds_dict[idx], 0) / 255
            gt = cv2.imread(gt_dict[idx], 0) / 255
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

            gt_bool = np.array(gt, dtype=bool).flatten()
            pred = pred.flatten()
            smooth = 1
            iou_th_dict = {}
            for th in thresholds:
                pred_th = pred > th

                mask1_area = np.count_nonzero(pred_th == 1)
                mask2_area = np.count_nonzero(gt_bool == 1)
                intersection = np.count_nonzero(np.logical_and(pred_th, gt_bool))
                iou_th = (intersection) / (
                    mask1_area + mask2_area - intersection + smooth
                )

                iou_th_dict[th] = iou_th
            iou_dat[idx] = iou_th_dict

    return get_max_iou(iou_dat)


def get_max_iou(iou_dat):
    th_list = list(iou_dat[list(iou_dat)[0]])
    iou_at_th = []
    for th in th_list:
        ioulist = [iou_dat[idx][th] for idx in iou_dat]
        val = sum(ioulist) / len(ioulist)
        if np.isnan(val):
            val = 0
        iou_at_th.append((val, th))

    iou_at_th = sorted(iou_at_th, key=lambda x: x[0])
    print(iou_at_th[-1][0], iou_at_th[-1][1])
    return {"iou": iou_at_th[-1][0], "iou_th": iou_at_th[-1][1]}
