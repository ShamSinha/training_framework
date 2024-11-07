from copy import deepcopy

import pyrootutils

root = pyrootutils.setup_root(
    search_from="./", indicator=[".git", "pyproject.toml"], pythonpath=True, dotenv=True
)

import os
from ast import literal_eval
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import typer
from loguru import logger
from omegaconf import OmegaConf
from qct_data.ct_loader import CTAnnotLoader
from qct_utils import ITKDIM, CtscanAnnot, sitk_mask2rle, sitk_rle2mask
from qct_utils.utils import transfer_sitk_params
from tqdm import tqdm

from src.apps.segmentation_infer import NoduleSegmentation
from src.metrics.segmentation.associate_annots import associate_annots

app = typer.Typer()


def _load_dataset(data: Union[List[str], List[pd.DataFrame]]) -> pd.DataFrame:
    """Data can be a list of dataframes or string paths.

    Returns single dataframe with all the datasets.
    """
    if isinstance(data, str) or isinstance(data, pd.DataFrame):
        data = [data]
    if isinstance(data[0], str):
        data = pd.concat([pd.read_csv(p) for p in data])
    elif isinstance(data[0], pd.DataFrame):
        data = pd.concat(data)
    return data


def get_nodules_based_on_chars(
    data: Union[str, pd.DataFrame, List[str], List[pd.DataFrame]],
    char_name: str,
    char_values: List[str],
):
    """
    Get all the nodules based on the mentioned characteristics.
    Args:
    -----
    data: dataset from which get the series ids from
    char_name: characteristics name
    char_values: which values to consider the nodules
    """
    df = _load_dataset(data)
    df = df[~df["meta"].isna()]  # filter where there is no meta
    df[char_name] = df["meta"].apply(lambda x: literal_eval(x)[char_name])
    df_filtered = df[df[char_name].isin(char_values)]
    return df_filtered["Nodule_name"].unique().tolist()


def get_nodules_based_on_multiple_chars(
    data: Union[str, pd.DataFrame, List[str], List[pd.DataFrame]],
    char_names: List[str],
    char_values: List[List[str]],
):
    nodule_ids = []
    for char_name, char_value in zip(char_names, char_values):
        nodule_ids.extend(get_nodules_based_on_chars(data, char_name, char_value))
    return list(set(nodule_ids))


def create_crops(
    scans_root: str,
    nodule_ids: List[str],
    data: Union[str, pd.DataFrame, List[str], List[pd.DataFrame]],
    save_root: str,
    z_margin: int = 10,
    annotated_nodule_ids: Optional[List[str]] = None,
    roi_margin: List[int] = (2, 5, 5),
) -> None:
    """
    Creates nodule crops and mask region of interest for lung nodules.
    Args:
    -----
    scans_root: root path to nifti files.
    nodule_ids: nodule ids to create the crops
    data: ctloader csv paths or DataFrames
    save_root: save root for the crops
    z_margin: z margin for the crops
    roi_margin: roi margin on top of the nodule bbox.
    Returns:
    --------
    None
    - Saves the `crops` and the `roi` into the `save_root`.
    """
    df = _load_dataset(data=data)
    nodule_ids = sorted(nodule_ids)
    logger.info(f"Total nodules {len(nodule_ids)}")
    series_id = None
    crops_metadata = {}
    os.makedirs(os.path.join(save_root, "roi"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "nodules"), exist_ok=True)
    ctloader_data = {"Nodule_name": [], "annot": [], "meta": [], "annotated_by": []}
    for nodule_id in tqdm(nodule_ids, desc="generating cache", colour="green"):
        if annotated_nodule_ids is not None and nodule_id in annotated_nodule_ids:
            continue
        if nodule_id.split("_")[0] != series_id:
            scan_path = os.path.join(scans_root, nodule_id.split("_")[0])
            scan_sitk = sitk.ReadImage(scan_path)
            scan = sitk.GetArrayFromImage(scan_sitk)
        nodule_annot = literal_eval(df[df["Nodule_name"] == nodule_id].annot.iloc[0])
        mask = sitk_rle2mask(nodule_annot["rle"], ITKDIM(**nodule_annot["size"]))
        zz, yy, xx = np.where(mask == 1)
        nodule_crop = scan[
            max([zz.min() - z_margin, 0]) : min([zz.max() + z_margin, scan.shape[0]]), ...
        ]
        mask[
            max([zz.min() - 2, 0]) : min([zz.max() + 2, mask.shape[0]]),
            yy.min() - 5 : yy.max() + 5,
            xx.min() - 5 : xx.max() + 5,
        ] = 1
        mask_crop = mask[
            max([zz.min() - z_margin, 0]) : min([zz.max() + z_margin, scan.shape[0]]), ...
        ]

        ctloader_data["Nodule_name"].append(nodule_id.replace("_", ".") + "_nodule")
        ctloader_data["annot"].append(
            {
                "size": dict(zip(["z", "y", "x"], nodule_crop.shape)),
                "bbox": {
                    "x_center": 1,
                    "y_center": 1,
                    "z_center": 1,
                    "w": 1,
                    "h": 1,
                    "d": 1,
                },  # dummy values since bbox should not be None
                "rle": sitk_mask2rle(mask_crop.astype(np.int8))["rle"],
            }
        )
        ctloader_data["meta"].append(None)
        ctloader_data["annotated_by"].append(None)

        crops_metadata[nodule_id] = {
            "z_crop_range": (int(zz.min()) - z_margin, int(zz.max()) + z_margin),
            "roi_range": {
                "z": (int(zz.min()) - roi_margin[0], int(zz.max()) + roi_margin[0]),
                "y": (int(yy.min()) - roi_margin[1], int(yy.max()) + roi_margin[1]),
                "x": (int(xx.min()) - roi_margin[2], int(xx.max()) + roi_margin[2]),
            },
        }
        sitk.WriteImage(
            transfer_sitk_params(mask_crop, scan_sitk),
            os.path.join(save_root, "roi", f"{nodule_id.replace('_', '.')}.nii.gz"),
        )
        sitk.WriteImage(
            transfer_sitk_params(nodule_crop, scan_sitk),
            os.path.join(save_root, "nodules", f"{nodule_id.replace('_', '.')}.nii.gz"),
        )

    logger.info("Crops and masks are saved.")
    pd.DataFrame(ctloader_data).to_csv(os.path.join(save_root, "ctloader.csv"), index=False)
    meta_info_path = os.path.join(save_root, "meta_info.yaml")
    OmegaConf.save(crops_metadata, meta_info_path)
    logger.info(
        f"Crops meta is saved at {meta_info_path}. use this to update the ctloader csv with the new annotations."
    )


def _get_dice(gt_annot: CtscanAnnot, pred_annot: CtscanAnnot) -> float:
    gt_mask = gt_annot.annot.get_mask()
    pred_mask = pred_annot.annot.get_mask()
    return float(2 * np.sum(gt_mask * pred_mask)) / (np.sum(gt_mask) + np.sum(pred_mask))


def filter_based_on_dice(
    ckpt_path: str,
    gt_csv: str,
    save_path: str,  # shoulr be a csv path
    scans_root: Optional[str] = None,
    pred_csv: Optional[str] = None,
    iou_thres: float = 0.1,
    considered_sids: List[str] = None,
    crop_margin: ITKDIM = ITKDIM(z=15, y=80, x=80),
    roi_margin: ITKDIM = ITKDIM(z=5, y=20, x=20),
    conf_thr: float = 0.5,
    device: torch.device = torch.device("cuda:0"),
):
    """Filter nodules based on Dice score.

    Use this to improve annotations done for detection where segmentation masks are not correct.
    """
    assert pred_csv is not None or scans_root is not None
    find_tps = True if pred_csv is not None else False
    gt_annotloader = CTAnnotLoader("", gt_csv, dummy_scans=True, series_ids=considered_sids)
    if not find_tps:
        seg_infer = NoduleSegmentation(checkpoint_path=ckpt_path, device=device)
        all_sids = gt_annotloader.scan_names
    else:
        pred_annotloader = CTAnnotLoader(
            "", pred_csv, dummy_scans=True, series_ids=considered_sids
        )
        all_sids = list(set(gt_annotloader.scan_names).intersection(pred_annotloader.scan_names))
    # find tp mappings in case of not mapped already
    dice_scores = []
    for sid in tqdm(all_sids, desc="processing series", colour="green"):
        gt_ctscan = gt_annotloader[sid]
        if find_tps:
            pred_ctscan = pred_annotloader[sid]
            gt_ctscan, pred_ctscan = associate_annots(gt_ctscan, pred_ctscan, iou_thr=iou_thres)
            for pred_annot in pred_ctscan.Annot:
                if pred_annot.pred_annot.TP_FP_FN == "TP":  # only consider the TP
                    gt_nodule_idx = pred_annot.pred_annot.gt_Nodule_name.split("_")[1]
                    gt_annot: CtscanAnnot = gt_ctscan.Annot[int(gt_nodule_idx)]
                    dice_scores.append(
                        {
                            "Nodule_name": sid + "_" + gt_annot.Nodule_name.split("_")[1],
                            "dice_score": _get_dice(gt_annot, pred_annot),
                        }
                    )
        else:
            scan = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(scans_root, f"{sid}.nii.gz"))
            )

            for gt_annot in gt_ctscan.Annot:
                pred_annot = seg_infer.predict_ctscan_annot(
                    ndimg=scan,
                    ctscanannot=deepcopy(gt_annot),
                    spacing=gt_ctscan.Spacing,
                    crop_margin=crop_margin,
                    roi_margin=roi_margin,
                    conf_thr=conf_thr,
                    volume_threshold=0,
                    store_volume_diameter=False,
                )
                dice_scores.append(
                    {
                        "Nodule_name": sid + "_" + gt_annot.Nodule_name.split("_")[1],
                        "dice_score": _get_dice(gt_annot, pred_annot),
                    }
                )

    df = pd.DataFrame(dice_scores)
    df.to_csv(save_path, index=True)
    logger.info(f"File saved at {save_path}")


def save_model(ckpt_path: str, save_path: str):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["state_dict"]

    prefix = "net."
    for key in list(state_dict.keys()):
        state_dict[key[key.startswith(prefix) and len(prefix) :]] = state_dict.pop(key)

    torch.save(state_dict, save_path)
    logger.info(f"Checkpoint saved at {save_path}")


if __name__ == "__main__":
    # scans_root = "/cache/datanas1/qct-nodules/nifti_with_annots/nlst"
    # data = "/home/users/souvik.mandal/projects/qct/qct_data_updates/annotations/2_stage_annotations/nlst_train/rad2_stage1.csv"
    # characteristics = ["Texture"]
    # char_values = [
    #     ["Solid/Mixed", "Part Solid/Mixed", "Non-Solid/GGO", "Non-Solid/Mixed"],
    #     # ["Marked Spiculation", "Near Marked Spiculation", "Medium Spiculation"]
    # ]
    # save_root = "/cache/expdata1/user_checkpoints/souvik.mandal/segm_cache/nlst_texture"
    # z_margin = 10
    # nodule_ids = get_nodules_based_on_multiple_chars(data, characteristics, char_values)[:30]

    # create_crops(scans_root, nodule_ids, data, save_root, z_margin=z_margin, annotated_nodule_ids=None)

    save_model(
        "/home/users/souvik.mandal/datasets/model_checkpoints/training/qCT_nodule_segm/runs/2023-03-02_08-36-11/checkpoints/epoch=186_step=18887_val_dice=0.8614.ckpt",
        "noisy_bbox_epoch=186_8614.ckpt",
    )
    filter_based_on_dice(
        ckpt_path="noisy_bbox_epoch=186_8614.ckpt",
        gt_csv="/home/users/souvik.mandal/projects/qct/qct_data_updates/data/studies/FDA/WCG/wcg_s1_s2_gt.csv",
        pred_csv=None,
        scans_root="/home/users/souvik.mandal/projects/qct/qct_data_updates/data/studies/FDA/WCG/data_tmp",
        iou_thres=0.1,
        save_path="very_bad_al_technique2.csv",
        considered_sids=[
            "1.3.6.1.4.1.55648.263191745543843798253004438085589084531.2",
            "1.3.6.1.4.1.55648.263191745543843798253004438085589084531.2",
            "1.3.6.1.4.1.55648.170728308031556976617895037945341854902.2",
        ],
    )
