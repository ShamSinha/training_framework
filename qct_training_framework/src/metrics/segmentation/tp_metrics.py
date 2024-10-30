import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from qct_data.ct_loader import CTAnnotLoader
from qct_utils.ct_schema import Ctannotdim, Ctscan, CtscanAnnot, MaskRle
from qct_utils.sitk_ops import sitk_rle2mask
from torchmetrics import Metric
from tqdm import tqdm

# from qct_data.associate_annots import associate_annots
from .associate_annots import associate_annots
from .dice import Dice3D
from .volume_error import VolumeError

logger = logging.getLogger(__name__)


char_cls_mappings = {
    "Texture": {
        "Non-Solid/GGO": 2,
        "non-Solid/Mixed": 2,
        "Non-Solid/Mixed": 2,
        "Solid": 0,
        "solid": 0,
        "Part Solid/Mixed": 1,
        "Solid/Mixed": 1,
        # self mappings
        "texture_non_solid": 2,
        "texture_solid": 0,
        "texture_part_solid": 1,
    },
    "Calcification": {
        "solid": 1,
        "Solid": 1,
        "Non-central": 1,
        "Central": 1,
        "Absent": 0,
        "absent": 0,
        "Fat": -1,
        # self mappings,
        "calcification_solid": 1,
        "calcification_non_central": 1,
        "calcification_central": 1,
        "calcification_absent": 0,
    },
    "Spiculation": {
        "No Spiculation": 0,
        "Nearly No Spiculation": 0,
        "Medium Spiculation": 1,
        "Marked Spiculation": 1,
        "Near Marked Spiculation": 1,
        "near Marked Spiculation": 1,
        "near Sharp": 1,
        # self mappings
        "spiculation_no": 0,
        "spiculation_medium": 1,
        "spiculation_marked": 1,
    },
}


def _get_metadata_cls(char, value):
    pass


class TPMetrics(Metric):
    def __init__(
        self,
        match_annots: bool = False,
        iou_thr: float = 0.1,
        box_conf_thr: float = 0.0,
        dice3d_kwargs: Dict = dict(conf_thresholds=[0.5], num_classes=1, ignore_background=False),
        volume_bins: List[int] = [-np.inf, 1, 2, np.inf],
        z_spacing_bins: List[int] = [-np.inf, 2, 4, np.inf],
    ) -> None:
        """This is useful when Dice3D needs to be calculated on only TPs found btw GT & Pred. This
        is different from Dice3D. If `match_annots`` is true then Assigns every box in
        gt_ctscan.Annot & pred_ctscan.Annot to TP, FP, FN. Only Dice is calculated for TP bboxes in
        pred_ctscan.

        Args:
            match_annots (bool, optional): Match annotations to assign TP, FP, FN. Defaults to False.
            iou_thr (float, optional): IOU thr when determining one bbox is Postive. Defaults to 0.3.
            box_conf_thr (float, optional): Remove boxes less than this threshold. Defaults to 0.0.
        """

        self.match_annots = match_annots
        self.iou_thr = iou_thr
        self.box_conf_thr = box_conf_thr
        self.metrics = [Dice3D(**dice3d_kwargs), VolumeError(cls_idx=0)]
        self.dice3d_kwargs = dice3d_kwargs
        self.volume_bins = volume_bins
        self.z_spacing_bins = z_spacing_bins
        self.gt_nods = []
        self.pred_nods = []

        self.total_gt_masks = 0
        self.total_pred_masks = 0
        self.total_tp_masks = 0
        self.total_fp_masks = 0
        self.total_fn_masks = 0
        self.meta_data = {
            "Texture": [],
            "Calcification": [],
            "Spiculation": [],
            "Volume": [],
            "z_spacing": [],
        }

    def update(
        self,
        pred_ctscans: Union[List[Ctscan], Ctscan],
        gt_ctscans: Union[List[Ctscan], Ctscan],
    ):
        """Assumes each of pred_ctscans: pred_ctscan.Annot[i].annot.bbox.

        Args:
            pred_ctscans (Union[List[Ctscan], Ctscan]): _description_
            gt_ctscans (Union[List[Ctscan], Ctscan]): _description_
        """
        if not isinstance(pred_ctscans, List):
            pred_ctscans = [pred_ctscans]
        if not isinstance(gt_ctscans, List):
            gt_ctscans = [gt_ctscans]

        for pred_ctscan, gt_ctscan in zip(pred_ctscans, gt_ctscans):
            if pred_ctscan.Annot is not None:
                pred_ctscan.Annot = [
                    i
                    for i in pred_ctscan.Annot
                    if i.annot.conf is None or i.annot.conf >= self.box_conf_thr
                ]
            self.total_gt_masks += len(gt_ctscan.Annot)
            self.total_pred_masks += len(pred_ctscan.Annot)
            if self.match_annots:
                gt_ctscan, pred_ctscan = associate_annots(
                    gt_ctscan, pred_ctscan, iou_thr=self.iou_thr
                )

            TP_annots, FP_annots, FN_annots = self._filter_annots(gt_ctscan, pred_ctscan)
            self.total_tp_masks += len(TP_annots)
            self.total_fp_masks += len(FP_annots)
            self.total_fn_masks += len(FN_annots)
            with open("check_list.csv", "a") as f:
                f.write(
                    f"{gt_ctscan.SeriesInstanceUID}, {len(TP_annots)},{len(FP_annots)},{len(FN_annots)}\n"
                )
            logger.info(
                f"sid: {gt_ctscan.SeriesInstanceUID} Associate annots: TP: {len(TP_annots)}, FP: {len(FP_annots)}, FN: {len(FN_annots)}"
            )

            gt_masks, pred_masks, volumes = [], [], []

            for gt_annot, pred_annot in TP_annots:
                gt_mask = self._get_mask(gt_annot)
                pred_mask = self._get_mask(pred_annot)
                gt_masks.append(torch.Tensor(gt_mask).unsqueeze(0))
                pred_masks.append(torch.Tensor(pred_mask).unsqueeze(0))
                volume = gt_mask.sum() * np.prod(gt_annot.annot.spacing.xyz)
                volumes.append(volume)
                gt_annot: CtscanAnnot = gt_annot
                # extract the metadata for subgroupanalysis
                if gt_annot.meta is None:
                    self.meta_data["Texture"].append(-1)
                    self.meta_data["Calcification"].append(-1)
                    self.meta_data["Spiculation"].append(-1)
                else:
                    self.meta_data["Texture"].append(
                        char_cls_mappings["Texture"].get(gt_annot.meta.Texture, -1)
                    )
                    self.meta_data["Calcification"].append(
                        char_cls_mappings["Calcification"].get(gt_annot.meta.Calcification, -1)
                    )
                    self.meta_data["Spiculation"].append(
                        char_cls_mappings["Spiculation"].get(gt_annot.meta.Spiculation, -1)
                    )
                self.meta_data["Volume"].append(
                    np.where(np.histogram(volume.item(), self.volume_bins)[0] == 1)[0][0]
                )
                self.meta_data["z_spacing"].append(
                    np.where(np.histogram(gt_annot.annot.spacing.z, self.z_spacing_bins)[0] == 1)[
                        0
                    ][0]
                )

            if len(gt_masks) > 0 and len(pred_masks) > 0:
                for metric in self.metrics:
                    if isinstance(metric, VolumeError):
                        metric.update(pred_masks, gt_masks, volumes)
                    else:
                        metric.update(pred_masks, gt_masks)

            self.gt_nods.extend(gt_masks)
            self.pred_nods.extend(pred_masks)

    def update_csv(self, pred_csv: str, gt_csv: str):
        # alternate update method that takes the gt and pred ctloader csv to
        # compute the dice score for the tp nodules
        # Total   GT: 610
        # Total Pred: 1281
        # Total   TP: 326
        # Total   FP: 955
        # Total   FN: 284
        # [tensor(0.6826)]
        pred_annot_loader = CTAnnotLoader(scans_root="", csv_loc=pred_csv, dummy_scans=True)
        gt_annot_loader = CTAnnotLoader(scans_root="", csv_loc=gt_csv, dummy_scans=True)
        gt_pos_sids = gt_annot_loader.scan_names.tolist()
        pred_pos_sids = pred_annot_loader.scan_names.tolist()
        all_series_ids = list(set(gt_pos_sids + pred_pos_sids))
        for sid in tqdm(all_series_ids[:40], leave=True, colour="green", desc="processing data"):
            if sid not in gt_pos_sids or sid not in pred_pos_sids:
                continue  # either not annots for gt or no predictions for this scan. Skip since no TP
            gtscan = gt_annot_loader[sid]
            predscan = pred_annot_loader[sid]
            self.update(pred_ctscans=[predscan], gt_ctscans=[gtscan])

    @staticmethod
    def _get_mask(ctscanannot: CtscanAnnot):
        annot_dim: Ctannotdim = ctscanannot.annot
        if annot_dim.mask is not None:
            return annot_dim.mask
        else:
            shape = annot_dim.size
            if isinstance(annot_dim.rle, MaskRle):
                return sitk_rle2mask(annot_dim.rle.rle, shape)
            return sitk_rle2mask(annot_dim.rle, shape)

    def _filter_annots(self, gt_ctscan: Ctscan, pred_ctscan: Ctscan):
        TP_annots = []
        FP_annots = []
        FN_annots = []
        if not self.match_annots:
            assert len(gt_ctscan.Annot) == len(
                pred_ctscan.Annot
            ), f"Given argument match_annots: {self.match_annots}, Assumes each gt_ctscan.Annot[idx] corresponds to pred_ctscan.Annot[idx]. But len(gt_ctscan.Annot): {len(gt_ctscan.Annot)} and len(pred_ctscan.Annot): {len(pred_ctscan.Annot)} are different. "
            return zip(gt_ctscan.Annot, pred_ctscan.Annot), [], []

        if pred_ctscan.Annot is not None:
            for ctscan_annot in pred_ctscan.Annot:
                if ctscan_annot.pred_annot.TP_FP_FN == "TP":
                    gt_nodule_name = ctscan_annot.pred_annot.gt_Nodule_name
                    for gt_ctscan_annot in gt_ctscan.Annot:
                        if gt_ctscan_annot.Nodule_name == gt_nodule_name:
                            TP_annots.append([gt_ctscan_annot, ctscan_annot])
                            break
                elif ctscan_annot.pred_annot.TP_FP_FN == "FP":
                    FP_annots.append(ctscan_annot)
                else:
                    raise Exception("TP/FP is not assigned")

        if gt_ctscan.Annot is not None:
            for ctscan_annot in gt_ctscan.Annot:
                if ctscan_annot.pred_annot.TP_FP_FN == "FN":
                    FN_annots.append(ctscan_annot)

        return TP_annots, FP_annots, FN_annots

    def _compute_metrics(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
        filtered_voxel_vols: List[torch.Tensor],
    ):
        """Add any new metrics here."""
        results = {}
        for metric in self.metrics:
            if isinstance(metric, Dice3D):
                dice_metric = Dice3D(**self.dice3d_kwargs)
                dice_metric.update(predictions, labels)
                dice_score = dice_metric.compute().item()
                sample_size = len(predictions)
                results.update({"Dice Score": dice_score, "Sample Size": sample_size})
            elif isinstance(metric, VolumeError):
                pred_volume = torch.tensor(
                    [
                        pred[0].sum().item() * vox_vol.item()
                        for pred, vox_vol in zip(predictions, filtered_voxel_vols)
                    ]
                )
                label_volume = torch.tensor(
                    [
                        pred[0].sum().item() * vox_vol.item()
                        for pred, vox_vol in zip(labels, filtered_voxel_vols)
                    ]
                )
                sample_size = len(labels)
                volume_error = torch.mean(
                    torch.abs(pred_volume - label_volume) / label_volume
                ).item()
                results.update({"Volume Error mean": volume_error, "Sample Size": sample_size})

            else:
                raise NotImplementedError(f"metric {metric} is not Implemented.")
        return results

    def _compute(self, meta_values, predictions, labels, voxel_vol):
        meta_values = (
            meta_values.clone().detach() if isinstance(meta_values, torch.Tensor) else meta_values
        )
        results = {}
        unique_meta_values = torch.unique(torch.tensor(meta_values)).tolist()
        for unique_key in unique_meta_values:
            indices = torch.where(torch.tensor(meta_values) == unique_key)[0].tolist()
            filtered_predictions = list(map(predictions.__getitem__, indices))
            filtered_labels = list(map(labels.__getitem__, indices))
            filtered_voxel_vols = list(map(voxel_vol.__getitem__, indices))
            results[unique_key] = self._compute_metrics(
                filtered_predictions, filtered_labels, filtered_voxel_vols
            )
        return results

    def _shared_computation(self):
        # returns results in following format
        # {"meta": {class_name: {metric_score: x, Sample Size: y}}}
        # metadata, predictions, labels = self._extract_data(outputs)
        voxel_vol = self.meta_data.pop("Volume")
        results = {}
        for metadata_key, metadata_values in tqdm(
            self.meta_data.items(),
            desc="Processing subgroup metadata",
            leave=False,
            total=len(self.meta_data),
        ):
            results[metadata_key] = self._compute(
                metadata_values, self.pred_nods, self.gt_nods, voxel_vol=voxel_vol
            )
        return results

    def compute(self):
        print(f"Total   GT: {self.total_gt_masks}")
        print(f"Total Pred: {self.total_pred_masks}")
        print(f"Total   TP: {self.total_tp_masks}")
        print(f"Total   FP: {self.total_fp_masks}")
        print(f"Total   FN: {self.total_fn_masks}")
        return {
            "Full-DS": [metric.compute() for metric in self.metrics],
            "Subgroup": self._shared_computation(),
        }


# if __name__=="__main__":
#     pred_csv = "/home/users/souvik.mandal/projects/qct/qct_data_updates/data/studies/FDA/WCG/wcg_s1_s2.csv"
#     gt_csv = "/home/users/souvik.mandal/projects/qct/qct_data_updates/data/studies/FDA/WCG/wcg_s1_s2_gt.csv"
#     dice_metric = TPDice3D()
#     dice_metric.update_csv(pred_csv=pred_csv, gt_csv=gt_csv)
#     print(dice_metric.compute())
