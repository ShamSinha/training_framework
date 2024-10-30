import hydra
import torch
from voxdet.metrics.det_metrics import DetMetrics, assign_tp_fp_fn_linear_assignment
import pandas as pd
import os
from loguru import logger
from voxdet.metrics.sub_level_analysis import convert2df

from copy import deepcopy
from torchvision.transforms import Compose

from omegaconf import OmegaConf, ListConfig
from tqdm.auto import tqdm
from voxdet.metrics.sub_level_analysis import convert2df
import numpy as np
from monai.data.box_utils import non_max_suppression
import argparse
import fastcore.all as fc


def to_list(transforms: ListConfig):
    return OmegaConf.to_container(transforms, resolve=True)


parser = argparse.ArgumentParser(description="Run inference on CTScan")

if __name__ == "__main__":

    parser.add_argument("--device", type=str, help="which device to use for inference")
    parser.add_argument("--phase", type=str, help="dataset status train/val/test")
    parser.add_argument("--roi", type=str, help="roi : individual/whole")
    parser.add_argument(
        "datasets", type=str, nargs="+", help="A list of datasets to be processed."
    )

    args = parser.parse_args()

    # Accessing the arguments
    datasets = args.datasets
    phase = args.phase
    roi = args.roi

    logger.debug(datasets)
    logger.debug(phase)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize("../hydra_configs/evaluation/", version_base="1.2")
    cfg = hydra.compose("retinanet_det.yaml")

    # ckpt_dict = {"IW_v2_all_dataset": "epoch=564_step=5085_val_mAP=0.530.ckpt"}
    ckpt_dict = {
        "IW_march": "epoch=564_step=5085_val_mAP=0.530.ckpt",
        "IW_april": "epoch=89_step=1530_val_mAR=0.641.ckpt",
        "IW_april_v2": "epoch=344_step=6210_val_mAP=0.463.ckpt",
    }

    model_to_use = ckpt_dict["IW_april"]
    cfg.model.checkpoint_path = (
        f"/cache/fast_data_nas8/qct/shubham/qct_det_ckpt/{model_to_use}"
    )

    cfg.model.device = args.device
    cfg.model.inf_safe = False

    model = hydra.utils.instantiate(cfg.model)
    froc_thresholds = hydra.utils.instantiate(cfg.froc_thresholds)

    for dataset in datasets:
        fc.Path.mkdir(
            fc.Path(f"/cache/fast_data_nas8/qct/shubham/det_prediction_calibration/{dataset}/"),
            exist_ok=True,
        )
        csv_loc = f"/home/users/shubham.kumar/projects/interns/qct_data/qct_data_status/det/data/{dataset}.csv"
        df = pd.read_csv(csv_loc)

        series_ids = df[df["status"] == phase]["scan_name"].unique()

        lung_mask_dir = (
            f"/cache/fast_data_nas8/qct/shubham/lung_mask_cache_det/{dataset}/"
        )

        # print(cfg.with_window_transform)

        with_window_transform = to_list(
            hydra.utils.instantiate(cfg.with_window_transform).transforms
        )

        if roi == "whole":
            cfg.complete_lung_transform.transforms[1].cache_dir = lung_mask_dir
            complete_lung_transform = to_list(
                hydra.utils.instantiate(cfg.complete_lung_transform).transforms
            )
            boxes_tf = Compose([complete_lung_transform[0]])

        if roi == "individual":
            cfg.left_lung_transform.transforms[1].cache_dir = lung_mask_dir
            cfg.right_lung_transform.transforms[1].cache_dir = lung_mask_dir

            left_lung_transform = to_list(
                hydra.utils.instantiate(cfg.left_lung_transform).transforms
            )
            right_lung_transform = to_list(
                hydra.utils.instantiate(cfg.right_lung_transform).transforms
            )
            boxes_tf = Compose([left_lung_transform[0]])

        for sid in tqdm(series_ids, desc=dataset):
            out_dict = {}
            # if os.path.exists(f"/cache/fast_data_nas8/qct/shubham/det_prediction/{dataset}/{sid}.pt") :
            #     continue
            if os.path.exists(
                f"/cache/fast_data_nas8/qct/shubham/det_ctscan/{dataset}/{sid}.pt"
            ):
                try:
                    img = torch.load(
                        f"/cache/fast_data_nas8/qct/shubham/det_ctscan/{dataset}/{sid}.pt",
                        map_location="cpu",
                    )
                    bx = torch.load(
                        f"/cache/fast_data_nas8/qct/shubham/det_gt_annot/{dataset}/{sid}.pt",
                        map_location="cpu",
                    )
                    print(bx.keys())

                    img["boxes"] = np.array(bx["bbox"])

                    gt_boxes = boxes_tf(img)["boxes"]

                    if roi == "individual":

                        model.transforms = Compose(
                            left_lung_transform + with_window_transform
                        )
                        left_img_out = model(deepcopy(img))
                        model.transforms = Compose(
                            right_lung_transform + with_window_transform
                        )
                        right_img_out = model(deepcopy(img))

                        img_out_boxes = np.vstack(
                            [left_img_out["boxes"], right_img_out["boxes"]]
                        )
                        img_out_scores = np.hstack(
                            [left_img_out["scores"], right_img_out["scores"]]
                        )
                        img_out_labels = np.hstack(
                            [left_img_out["labels"], right_img_out["labels"]]
                        )

                        # final NMS
                        keep = non_max_suppression(
                            img_out_boxes, img_out_scores, nms_thresh=0.1
                        )
                        new_boxes, new_scores = (
                            img_out_boxes[keep],
                            img_out_scores[keep],
                        )
                        # final CNF_thr

                    if roi == "whole":
                        model.transforms = Compose(
                            complete_lung_transform + with_window_transform
                        )

                        # model.model.set_sliding_window_inferer(
                        #     roi_size=list(img["images"].shape),
                        #     sw_batch_size=1,
                        #     overlap=0.5,
                        #     mode="constant",
                        #     cval=0,
                        #     padding_mode="constant",
                        #     sw_device=None,
                        #     device=None,
                        #     progress=False,
                        # )

                        complete_img_out = model(deepcopy(img))

                        new_boxes = complete_img_out["boxes"]
                        new_scores = complete_img_out["scores"]

                    # print(new_scores)

                    # keep = new_scores >= 0.9
                    # final_boxes, final_scores = new_boxes[keep], new_scores[keep]

                    out_dict["model"] = model_to_use
                    out_dict["gt_boxes"] = gt_boxes
                    out_dict["pred_scores"] = new_scores
                    out_dict["pred_boxes"] = new_boxes

                    

                    # tp, fp, fn, tp_iou = assign_tp_fp_fn_linear_assignment(
                    #     new_boxes, gt_boxes, 0.1
                    # )

                    # out_dict["tp"] = tp
                    # out_dict["fp"] = fp
                    # out_dict["fn"] = fn
                    # out_dict["tp_iou"] = tp_iou

                    torch.save(
                        out_dict,
                        f"/cache/fast_data_nas8/qct/shubham/det_prediction_calibration/{dataset}/{sid}.pt",
                    )

                except KeyboardInterrupt:
                    break

                except Exception as e:
                    logger.debug(e)
                    continue