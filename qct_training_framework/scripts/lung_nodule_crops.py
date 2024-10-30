import math
import os
from ast import literal_eval
from shutil import rmtree
from typing import List

import numpy as np
import pandas as pd
import SimpleITK as sitk
import typer
from loguru import logger
from qct_utils import ITKDIMFLOAT, sitk_rle2mask
from tqdm import tqdm

app = typer.Typer()


def xcyczc2xyz(bbox):
    xleft, xright = bbox["x_center"] - (bbox["w"] / 2), bbox["x_center"] + (bbox["w"] / 2)
    yleft, yright = bbox["y_center"] - (bbox["h"] / 2), bbox["y_center"] + (bbox["h"] / 2)
    zleft, zright = bbox["z_center"] - (bbox["d"] / 2), bbox["z_center"] + (bbox["d"] / 2)
    return [
        int(xleft),
        int(yleft),
        int(zleft),
        math.ceil(xright),
        math.ceil(yright),
        math.ceil(zright),
    ]


@app.command("dehanet_crops", help="create dehanet training cache.")
def generate_lung_nodule_crops_dehanet(
    scan_root: str = typer.Argument(..., help="Nifti file root folder."),
    csv_paths: List[str] = typer.Argument(..., help="ctloader csvs paths"),
    dataset_source: str = typer.Argument(..., help="dataset source eg. lidc, dsb"),
    cache_save_root: str = typer.Argument(..., help="root folder to save the cache"),
    z_margin: int = typer.Argument(..., help="nodule crop z margin"),
    x_margin: int = typer.Argument(..., help="nodule crop x margin"),
    y_margin: int = typer.Argument(..., help="nodule crop y margin"),
    roi_margin=typer.Argument(
        ..., help="roi margin on top of bbox. Should he a list in [z, y, x] format"
    ),
    dataset_split: str = typer.Argument(..., help="which dataset split train, test or val"),
    overwrite: bool = typer.Option(
        default=True, help="If true delete the old folder and overwrite the new data"
    ),
):
    roi_margin = literal_eval(roi_margin)
    cache_save_root = os.path.join(cache_save_root, dataset_split)
    if not overwrite and os.path.exists(cache_save_root):
        logger.info("Folder exists. Set `overwrite` to True to overwrite")
    elif overwrite and os.path.exists(cache_save_root):
        rmtree(cache_save_root)
    os.makedirs(cache_save_root, exist_ok=False)
    df = pd.concat([pd.read_csv(p) for p in csv_paths])
    logger.info(f"Total nodules count {len(df)}")
    data_dict = dict(
        nodule_name=[],
        crop_path=[],
        mask_path=[],
        roi_path=[],
        z_spacing=[],
        Texture=[],
        Calcification=[],
        Spiculation=[],
        volume=[],
        source=[],
        voxel_vol=[],
    )
    for _, row in tqdm(df.iterrows(), desc="Creating cache", total=len(df), colour="green"):
        nodule_name = row["Nodule_name"]
        annot = literal_eval(row["annot"])
        bbox = xcyczc2xyz(annot["bbox"])
        scan_path = os.path.join(scan_root, nodule_name.split("_")[0] + ".nii.gz")
        try:
            meta = literal_eval(row["meta"])
        except:  # noqa
            # NLST and some other dataset sometimes have None, nan as string.
            meta = -1
        z_spacing = (
            -1
            if meta == -1
            else (annot["spacing"]["z"] if "spacing" in annot and annot["spacing"] else None)
        )
        Texture = -1 if meta == -1 else meta["Texture"]
        Calcification = -1 if meta == -1 else meta["Calcification"]
        Spiculation = -1 if meta == -1 else meta["Spiculation"]
        gt_mask = sitk_rle2mask(
            annot["rle"]["rle"] if not isinstance(annot["rle"], tuple) else annot["rle"],
            ITKDIMFLOAT(**annot["size"]),
        )
        scan = sitk.GetArrayFromImage(sitk.ReadImage(scan_path))
        sz, sy, sx = scan.shape
        z1 = max(0, int(bbox[2]) - z_margin)
        z2 = min(sz, math.ceil(bbox[5]) + 1 + z_margin)
        y1 = max(0, int(bbox[1]) - y_margin)
        y2 = min(sy, math.ceil(bbox[4]) + 1 + y_margin)
        x1 = max(0, int(bbox[0]) - x_margin)
        x2 = min(sx, math.ceil(bbox[3]) + 1 + x_margin)

        scan_crop = scan[z1:z2, y1:y2, x1:x2]

        gt_mask_crop = gt_mask[z1:z2, y1:y2, x1:x2]
        # zz, yy, xx = np.where(gt_mask_crop == 1)
        roi_mask = np.zeros_like(gt_mask)
        roi_mask[
            max(0, int(bbox[2]) - roi_margin[0]) : min(sz, math.ceil(bbox[5]) + 1 + roi_margin[0]),
            max(0, int(bbox[1]) - roi_margin[1]) : min(sy, math.ceil(bbox[4]) + 1 + roi_margin[1]),
            max(0, int(bbox[0]) - roi_margin[2]) : min(sx, math.ceil(bbox[3]) + 1 + roi_margin[2]),
        ] = 1.0
        roi_mask = roi_mask[z1:z2, y1:y2, x1:x2]

        spacing = (
            np.prod(list(annot["spacing"].values()))
            if "spacing" in annot and annot["spacing"]
            else -1
        )
        volume = gt_mask_crop.sum() * spacing if spacing != -1 else -1

        # save the crops
        nodule_crop_root = os.path.join(cache_save_root, "nodule_crops")
        mask_crop_root = os.path.join(cache_save_root, "mask_crops")
        roi_crops_root = os.path.join(cache_save_root, "roi_crops")
        _ = [
            os.makedirs(p, exist_ok=True)
            for p in [nodule_crop_root, mask_crop_root, roi_crops_root]
        ]
        np.save(os.path.join(nodule_crop_root, f"{nodule_name}.npy"), scan_crop)
        np.save(os.path.join(mask_crop_root, f"{nodule_name}.npy"), gt_mask_crop)
        np.save(os.path.join(roi_crops_root, f"{nodule_name}.npy"), roi_mask)
        # update the meta info
        data_dict["Calcification"].append(Calcification)
        data_dict["Texture"].append(Texture)
        data_dict["Spiculation"].append(Spiculation)
        data_dict["z_spacing"].append(z_spacing)
        data_dict["nodule_name"].append(nodule_name)
        data_dict["volume"].append(volume)
        data_dict["source"].append(dataset_source)
        data_dict["voxel_vol"].append(spacing)
        data_dict["crop_path"].append(os.path.join(nodule_crop_root, f"{nodule_name}.npy"))
        data_dict["mask_path"].append(os.path.join(mask_crop_root, f"{nodule_name}.npy"))
        data_dict["roi_path"].append(os.path.join(roi_crops_root, f"{nodule_name}.npy"))

    meta_df = pd.DataFrame(data_dict)
    meta_df.to_csv(
        os.path.join(cache_save_root, f"{dataset_source}_{dataset_split}.csv"), index=False
    )
    logger.info(f"Dataset saved at {cache_save_root}")


if __name__ == "__main__":
    app()
