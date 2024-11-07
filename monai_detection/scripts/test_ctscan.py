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
    # parser.add_argument("--roi", type=str, help="roi : individual/whole")
    parser.add_argument(
        "datasets", type=str, nargs="+", help="A list of datasets to be processed."
    )

    args = parser.parse_args()

    # Accessing the arguments
    datasets = args.datasets
    phase = args.phase
    roi = "whole"

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
        "IW_may": "epoch=14_step=285_val_mAP=0.429.ckpt",
        "IW_june": "epoch=434_step=8265_val_mAP=0.419.ckpt", # diff windowing
        "IW_may_v2": "epoch=24_step=475_val_mAR=0.774.ckpt"
    }

    model_to_use = ckpt_dict["IW_may"]
    cfg.model.checkpoint_path = (
        f"/cache/fast_data_nas8/qct/shubham/qct_det_ckpt/{model_to_use}"
    )

    cfg.model.device = args.device
    cfg.model.inf_safe = False

    model = hydra.utils.instantiate(cfg.model)
    froc_thresholds = hydra.utils.instantiate(cfg.froc_thresholds)

    meters = [
        DetMetrics(iou_thr=j, conf_thr=i, froc_thresholds=froc_thresholds.tolist())
        for j in cfg.iou_thr
        for i in [0.5]
    ]

    for dataset in datasets:
        fc.Path.mkdir(
            fc.Path(f"/cache/fast_data_nas8/qct/shubham/det_pred_tempscale/{dataset}/"),
            exist_ok=True,
        )
        for meter in meters:
            meter.reset()
        csv_loc = f"/home/users/shubham.kumar/projects/qct_data/qct_data_status/det/data/{dataset}.csv"
        df = pd.read_csv(csv_loc)

        series_ids = df[df["status"] == phase]["scan_name"].unique()
            
        # csv_loc = "/home/users/shubham.kumar/projects/recist/recist_annotation_batch1_6_v2.csv"
        # df = pd.read_csv(csv_loc)
        # sids = df.annot_id.values
        # series_ids = np.unique([sid.rsplit("_")[0]  for sid in sids])


        lung_mask_dir = (
            f"/cache/fast_data_nas8/qct/shubham/lung_mask_cache_det/{dataset}/"
        )

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
                    # print(bx.keys())

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
                        # If we want to infer whole scan at once without providing any roi_size
                        
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

                    keep = new_scores >= 0.0
                    final_boxes, final_scores = new_boxes[keep], new_scores[keep]

                    # out_dict["model"] = model_to_use
                    out_dict["gt_boxes"] = gt_boxes
                    out_dict["pred_scores"] = final_scores
                    out_dict["pred_boxes"] = final_boxes

                    # tp, fp, fn, tp_iou = assign_tp_fp_fn_linear_assignment(
                    #     final_boxes, gt_boxes, 0.1
                    # )

                    # print(tp)
                    # print(fp)

                    # out_dict["tp"] = tp
                    # out_dict["fp"] = fp
                    # out_dict["fn"] = fn
                    # out_dict["tp_iou"] = tp_iou


                    if os.path.exists(f"/cache/fast_data_nas8/qct/shubham/det_pred_tempscale/{dataset}/{sid}.pt"):
                        new_dict = torch.load(f"/cache/fast_data_nas8/qct/shubham/det_pred_tempscale/{dataset}/{sid}.pt")
                    else:
                        new_dict = {}

                    new_dict[cfg.model.checkpoint_path] = out_dict

                    torch.save(
                        new_dict,
                        f"/cache/fast_data_nas8/qct/shubham/det_pred_tempscale/{dataset}/{sid}.pt",
                    )

                    for meter in meters:
                        meter.update(
                            torch.from_numpy(final_boxes),
                            torch.from_numpy(final_scores),
                            torch.from_numpy(gt_boxes),
                        )

                except KeyboardInterrupt:
                    break

                except Exception as e:
                    logger.debug(e)
                    continue

        metrics = [i.compute() for i in meters]
        df_metrics = convert2df(metrics)
        df_metrics.to_csv(f"evaluation_df_may/tempscale_{dataset}_{phase}.csv", index=False)
        # df_metrics.to_csv(f"evaluation_df_april/{dataset}_{phase}.csv", index=False)
