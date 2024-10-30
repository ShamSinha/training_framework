import hydra
import fastcore.all as fc 
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf, ListConfig

from qct_utils.ctscan.ct_loader import CTAnnotLoader

from voxdet.metrics.det_metrics import DetMetrics
from voxdet.metrics.sub_level_analysis import convert2df
from voxdet.create_ds import ctscan2dict

from torchvision.transforms import Compose
from monai.data.box_utils import non_max_suppression

def to_list(transforms: ListConfig):
    return OmegaConf.to_container(transforms, resolve=True)


def get_model(cfg, ckpt_path , device):
    cfg.model.checkpoint_path = ckpt_path

    cfg.model.device = device
    model = hydra.utils.instantiate(cfg.model)
    froc_thresholds = hydra.utils.instantiate(cfg.froc_thresholds)

    boxes_tf = Compose([left_lung_transform[0]])

    meters = [
        DetMetrics(iou_thr=j, conf_thr=i, froc_thresholds= froc_thresholds.tolist())
        for j in cfg.iou_thr
            for i in cfg.conf_thr
    ]

    return model, meters, boxes_tf
            
def get_df_metrics(ds : CTAnnotLoader, model, img_tf, boxes_tf ,meters, individual_lung: bool):
    for idx in tqdm(range(len(ds))):
        ctscan = ds[idx]
        for annot in ctscan.nodules:
            annot.gt.annot.bbox = annot.gt.annot.mask.to_boxc()
        img = ctscan2dict(ctscan)
        
        gt_boxes = boxes_tf(img)["boxes"]
        del img["boxes"]

        if individual_lung :
            model.transforms = Compose(left_lung_transform + img_tf)
            left_img_out = model(deepcopy(img))
            model.transforms = Compose(right_lung_transform + img_tf)
            right_img_out = model(deepcopy(img))
            
            img_out_boxes = np.vstack([left_img_out["boxes"], right_img_out["boxes"]])
            img_out_scores = np.hstack([left_img_out["scores"], right_img_out["scores"]])
            
            # final NMS
            keep = non_max_suppression(img_out_boxes, img_out_scores, nms_thresh=0.2)
            new_boxes, new_scores = img_out_boxes[keep], img_out_scores[keep]
            # final CNF_thr
            keep = new_scores > 0.05
            final_boxes, final_scores = new_boxes[keep], new_scores[keep]

        else :
            model.transforms = Compose(complete_lung_transform + img_tf)
            img_out = model(deepcopy(img))
            final_boxes = img_out["boxes"]
            final_scores = img_out["scores"]
                
        for meter in meters: meter.update(final_boxes, final_scores, gt_boxes)

    metrics = [i.compute() for i in meters]
    return convert2df(metrics)


if __name__ == "__main__":
    
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize("../hydra_configs/evaluation/", version_base="1.2")
    cfg = hydra.compose("retinanet_det.yaml")

    with_window_transform = to_list(hydra.utils.instantiate(cfg.with_window_transform).transforms)
    without_window_transform = to_list(hydra.utils.instantiate(cfg.without_window_transform).transforms)

    complete_lung_transform = to_list(hydra.utils.instantiate(cfg.complete_lung_transform).transforms)

    left_lung_transform = to_list(hydra.utils.instantiate(cfg.left_lung_transform).transforms)
    right_lung_transform = to_list(hydra.utils.instantiate(cfg.right_lung_transform).transforms)


    image_transforms_dict = {"IW" : with_window_transform,
                "I" : without_window_transform,
                "C" : without_window_transform,
                "CW": with_window_transform}

    individual_lung_dict = {"IW" : True , "I" : True , "C" : False , "CW" : False}

    ckpt_dict  = {"IW" : "epoch=414_step=1988_val_mAP=0.647.ckpt",
                "I" : "epoch=304_step=1221_val_mAP=0.662.ckpt",
                "C" : "epoch=595_step=2425_val_mAP=0.667.ckpt",
                "CW": "epoch=499_step=1500_val_mAP=0.647.ckpt"}

    datasets = ["lidc_1reader" , "lidc_3reader"]
    csv_loc = ["../studies/only_lidc/gt_csvs/lidc_union_1.csv", "../studies/only_lidc/gt_csvs/lidc_union_3.csv"]
    meta_root = "/cache/fast_data_nas72/qct/data_governance/series_dicts/lidc.pt"
    series_ids = pd.read_csv(fc.Path("../studies/only_lidc/folds1/folds_4.csv"))["scans"].values

    for k , _ in tqdm(ckpt_dict.items()) :
        model_to_use = ckpt_dict[k]
        ckpt_path = f"/cache/fast_data_nas8/qct/shubham/qct_det_ckpt/{model_to_use}"
        img_tf = image_transforms_dict[k]
        individual_lung = individual_lung_dict[k]
        model , meters , boxes_tf = get_model(cfg , ckpt_path , "cuda:1")

        logger.debug(f"ckpt_path: {ckpt_path}")
        logger.debug(f"individual_lung: {individual_lung}")

        for i in range(len(csv_loc)) : 
            dataset_name = datasets[i]
            logger.debug(f"dataset : {dataset_name}" )
            ds = CTAnnotLoader(scans_root=meta_root, csv_loc= csv_loc[i], series_ids = series_ids)
            for meter in meters:  meter.reset()

            df = get_df_metrics(ds, model, img_tf, boxes_tf, meters,individual_lung)
            df.to_csv(f"/home/users/shubham.kumar/projects/qct_nodule_detection/evaluation_df/{k}_{dataset_name}.csv" , index= False)
            logger.debug(f"{k}_{dataset_name}.csv saved " )



