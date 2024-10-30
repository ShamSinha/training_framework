from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.metrics import auc, roc_curve
from loguru import logger


def plot_auc_curve(gt_class_idx_l: np.ndarray, pred_class_prob_l: np.ndarray, n_classes: int):
    """AUC curve per class and macro average.

    Micro avagage does take care of imbalance, so not using it.
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(gt_class_idx_l == i, pred_class_prob_l[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # plot
    fig = plt.figure()
    # canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    lw = 2
    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label=f"ROC curve of class {i} (area = {roc_auc[i]:0.2f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC")
    plt.legend(loc="lower right")
    
    logger.debug("canvas")
    logger.debug(plt)

    return plt
