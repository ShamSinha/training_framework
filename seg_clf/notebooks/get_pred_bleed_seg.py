import pyrootutils
import hydra
import torch
import pandas as pd
from tqdm.auto import tqdm
from src.common.nn_modules.nets.segmentation.multitaskfusion import (
    MultiTaskFusionNet,
    modify_model_state_dict,
)
from torch.utils.data import DataLoader
from loguru import logger
import numpy as np


root = pyrootutils.setup_root(
    search_from="./", indicator=[".git", "pyproject.toml"], pythonpath=True, dotenv=True
)

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize("../configs/datamodule/", version_base="1.2")
data_module = hydra.compose("bleed_seg_datamodule.yaml")

datamodule = hydra.utils.instantiate(data_module)
test_ds = datamodule.test_ds
train_ds = datamodule.train_ds
val_ds = datamodule.val_ds


# ckpt_path = "/data_nas5/qer/shubham/ich_checkpoints/ich_segmentation/runs/2023-09-23_20-20-59/checkpoints/epoch=155_step=3744_val_loss=8.24.ckpt"
# ckpt_path = "/data_nas5/qer/shubham/ich_checkpoints/ich_segmentation/runs/2023-09-25_23-46-37/checkpoints/epoch=194_step=4680_val_loss=0.42.ckpt"
# ckpt_path = "/data_nas5/qer/shubham/ich_checkpoints/ich_segmentation/runs/2023-09-27_14-06-10/checkpoints/epoch=226_step=5448_val_loss=10.60.ckpt"
# ckpt_path = "/data_nas5/qer/shubham/ich_checkpoints/ich_segmentation/runs/2023-10-03_16-36-52/checkpoints/epoch=231_step=5668_val_loss=0.42.ckpt"
ckpt_path = "/data_nas5/qer/shubham/ich_checkpoints/ich_segmentation/runs/2023-10-03_22-00-45/checkpoints/epoch=235_step=5673_val_loss=0.49.ckpt"
k = torch.load(ckpt_path, map_location="cpu")
state_dict = k["state_dict"]
new_state_dict = modify_model_state_dict(state_dict, "net.", "")

model = MultiTaskFusionNet()

model.load_state_dict(new_state_dict)
model.to("cuda:2")
model.eval()

activation = torch.nn.Softmax(dim=1)

# def dice_coefficient(mask1, mask2):
#     # Ensure that both masks have the same shape
#     if mask1.shape != mask2.shape:
#         raise ValueError("Input masks must have the same shape.")

#     # Calculate the intersection and union of the two masks
#     intersection = np.sum(mask1 * mask2)
#     union = np.sum(mask1) + np.sum(mask2)

#     # Calculate the Dice coefficient
#     dice = (2.0 * intersection) / (
#         union + 1e-8
#     )  # Adding a small epsilon to avoid division by zero

#     return dice

from typing import Any, Dict, List, Optional, Literal

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from loguru import logger
from monai.networks import one_hot

class Dice3D(Metric):
    dice_scores: List[Tensor]
    full_state_update: bool = True

    def __init__(
        self,
        num_classes: int,
        conf_thresholds: List[float] = [0.5],
        compute_on_step: Optional[bool] = None,
        ignore_background: bool = False,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        **kwargs: Dict[str, Any]
    ):
        """_summary_

        Args:
            conf_thresholds (Optional[List[float]], optional): _description_. Defaults to None.
            compute_on_step (Optional[bool], optional): _description_. Defaults to None.
            ignore_background (bool): ignore dice computation for background pixels

        Example:
        ```
        import torch
        pred = [torch.ones(2, 3, 3, 3)]
        gt = [torch.cat([torch.zeros(1, 3, 3, 3), torch.ones(1, 3, 3, 3)], dim=0)]
        dice_mat = Dice3D(ignore_background=False)
        dice_mat.update(preds=pred, target=gt)
        print(dice_mat.compute())
        ```
        """
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        self.num_classes = num_classes
        self.conf_thresholds = conf_thresholds
        self.ignore_background = ignore_background
        self.average = average
        self.add_state("dice_scores", default=[], dist_reduce_fx=None)

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:  # type: ignore
        """Add detections and ground truth to the metric.
        Args:
            preds: A list consisting of Tensors. Each Tensor are prediction corresponding to one image.
            - Each probability Tensor should be in following format [num_class, z, y, x]
            - Each probability Tensors value range should be in [0, 1]
            target: A list consisting of tensors, Each Tensor are prediction corresponding to one image.
            - Each Tensor should be in following format [num_class, z, y, x]
            - values should be either 1 or 0
        """
        # _input_validator(preds, target, self.num_classes)

        if target[-1].shape[0] == 1 :
            target = one_hot(target,self.num_classes)
            
        assert target.shape == preds.shape

        self.input_device = preds.device
        self.dice_scores.append(self._get_intermediate_dice(preds, target))

    def _get_intermediate_dice(
        self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]
    ):
        nb_classes = len(self._get_classes())
        nb_confs = len(self.conf_thresholds)
        nb_imgs = len(preds)
        dice_score = -1 * torch.ones(nb_imgs, nb_classes, nb_confs).to(self.input_device)
        iou_score = -1 * torch.ones(nb_imgs, nb_classes, nb_confs).to(self.input_device)

        # logger.debug(f"nb_classes: {nb_classes}, nb_confs: {nb_confs}, nb_imgs: {nb_imgs}")
        # logger.debug(f"dice_score: {dice_score.shape}, iou_score: {iou_score.shape}")
        
        for idx_cls in range(nb_classes):
            if self.ignore_background and idx_cls == 0:
                continue
            for idx_conf in range(nb_confs):
                dice_score, iou_score = self._calculate_dice_score(
                    dice_score=dice_score,
                    iou_score=iou_score,
                    class_idx=idx_cls,
                    conf_idx=idx_conf,
                    pred_probs=preds,
                    gt_labels=target,
                )
        return dice_score

    def _get_classes(self):
        if self.num_classes is not None:
            return np.arange(self.num_classes)
        return []

    @staticmethod
    def __compute_dice(pred_mask, gt_mask):
        """
        Compute dice score given two binary mask
        dice score = 2*area(inter(pred_mask, gt_mask)) / area(gt_mask) + area(pred_mask)
        """
        return (2 * torch.sum(pred_mask * gt_mask)) / (torch.sum(pred_mask) + torch.sum(gt_mask))

    @staticmethod
    def __compute_iou(pred_mask, gt_mask):
        """Compute IoU of the predicted binary mask and ground truth mask."""

        return torch.sum(pred_mask * gt_mask) / torch.sum(torch.logical_or(pred_mask, gt_mask))

    def _calculate_dice_score(
        self,
        dice_score,
        iou_score,
        class_idx,
        conf_idx,
        pred_probs: List[Tensor],
        gt_labels: List[Tensor],
    ):
        for index, (detection_prob, gt_mask) in enumerate(zip(pred_probs, gt_labels)):
            det_binary = detection_prob[class_idx] >= self.conf_thresholds[conf_idx]
            gt_binary = gt_mask[class_idx]
            dice_score[index, class_idx, conf_idx] = self.__compute_dice(det_binary, gt_binary)
            iou_score[index, class_idx, conf_idx] = self.__compute_iou(det_binary, gt_binary)

        return dice_score, iou_score

    def compute(self):
        if self.dice_scores == []:
            return None
        dice_scores = torch.cat(self.dice_scores, dim=0)
        if self.average == "micro":
            return dice_scores[(dice_scores >= 0) & (~dice_scores.isnan())].mean()
        elif self.average == "macro":
            return (torch.mean(dice_scores , 0)).mean()
        elif self.average in ['none' , None] :
            return torch.mean(dice_scores , 0)[:,0]
        


def get_sensitivity(mask1, mask2):
    # Calculate True Positives (TP) and False Negatives (FN)
    TP = np.sum((mask1 == 1) & (mask2 == 1))
    FN = np.sum((mask1 == 1) & (mask2 == 0))

    # Calculate sensitivity (True Positive Rate)
    sensitivity = TP / (TP + FN)

    return sensitivity


def get_specificity(mask1, mask2):
    TN = np.sum((mask1 == 0) & (mask2 == 0))
    FP = np.sum((mask1 == 0) & (mask2 == 1))

    # Calculate specificity
    specificity = TN / (TN + FP)

    return specificity

dice = Dice3D(num_classes=2,average="micro",ignore_background=True).to("cuda:2")

li = []

for i in tqdm(range(len(test_ds))):
    try:
        data = test_ds[i]
        
        study_uid = data["study_uid"]
        gt = data["mask"]
        gt_arr = gt.squeeze().numpy()

        if np.max(gt_arr) == 0 :
            continue

        image = data["image"].unsqueeze(0).to("cuda:2")
        z_size = image.size(2)
        activation = torch.nn.Softmax(dim=1)
        model_outputs = model(image)
        soft_out = activation(model_outputs["student_mask"])
        soft_out_arr = soft_out.squeeze(0).detach().cpu().numpy()

        soft_teacher_out = activation(model_outputs["teacher_mask"])
        soft_teacher_out_arr = soft_teacher_out.squeeze(0).detach().cpu().numpy()
        

        
        dice.update(soft_teacher_out, gt.unsqueeze(0).to("cuda:2"))
        teacher_dsc = dice.compute()
        dice.dice_scores = []

        dice.update(soft_out, gt.unsqueeze(0).to("cuda:2"))
        dsc = dice.compute()
        dice.dice_scores = []

        teacher_pred = soft_teacher_out_arr[1] >= 0.5
        # teacher_dsc = dice_coefficient(gt, teacher_pred)
        teacher_sens = get_sensitivity(gt_arr, teacher_pred)
        teacher_spec = get_specificity(gt_arr, teacher_pred)

        pred = soft_out_arr[1] >= 0.5
        # dsc = dice_coefficient(gt, pred)
        sens = get_sensitivity(gt_arr, pred)
        spec = get_specificity(gt_arr, pred)

        li.append({"StudyUID": study_uid, "dsc": dsc.item(), "sens": sens, "spec": spec , "teacher_dsc": teacher_dsc.item(),
                    "teacher_sens" : teacher_sens, "teacher_spec": teacher_spec})
        # logger.debug(li[-1])

    except KeyboardInterrupt:
        break
    except Exception as e:
        logger.debug(e)
        continue

df = pd.DataFrame(li)
print(df)
df.to_csv("/home/users/shubham.kumar/projects/qct_training_framework/notebooks/new_quant_model_test_inference_v28.csv", index=False)


