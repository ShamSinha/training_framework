import os
import math
import pandas as pd
from loguru import logger
from typing import Dict, List, Optional, Union
from tqdm.auto import tqdm
import SimpleITK as sitk
import numpy as np
import h5py

def icv_vt_data_loader(
    csvs: List[str],
    data_root: List[Dict],
    sample_frac: Optional[Union[float, List[float]]],
    sample_size: Optional[Union[int, List[int]]],
    random_state: int,
    invert_scan: List[bool],
    # ext: List[str],
    load_scan: bool,
    keep_lung_only: bool,
    num_slices_per_volume: int,
    stride: int,
    phase: str,
    dataset : str, 

):
    """
    Base extraction function for Lung and Lobe segmentation.
    Args:
    -----
        csvs: List of csvs containing the data.
        data_root: Root directory of the data.
        num_slices_per_volume: Number of slices per volume.
        stride: Stride for the sliding window.
    """

    li = []

    for i in range(len(csvs)):
        csv = csvs[i]
        df = pd.read_csv(csv)
        df = df[df.Status == phase]
        frac = sample_frac[i]
        # logger.debug(f"Loading {phase} data from {csv} with frac {frac} and size {sample_size}.")
        df = df.sample(n = sample_size , frac = frac , axis = 0 , random_state = random_state)
        if dataset == "hdf5" : 
            logger.info(f"{phase} {csv}")
            li.extend(_load_hdf5_seg_dataset_ctloader(df, data_root[i], phase, num_slices_per_volume, stride, keep_lung_only, invert_scan[i], load_scan))
        elif dataset == "dcm" :
            li.extend(_load_seg_dataset_ctloader(df, data_root[i], num_slices_per_volume, stride, phase, keep_lung_only, invert_scan[i], load_scan))
    return li

def _load_hdf5_seg_dataset_ctloader(
    df: pd.DataFrame, data_root: Dict , phase: str, num_slices_per_volume: int, stride: int, keep_lung_only: bool, invert_ct: bool, load_scan: bool):
    data_dict = []
    df_boundary_dsc_lung_lobe = pd.read_csv("/home/users/shubham.kumar/projects/lung_lobe_segmentation/boundary_lung_lobe_dsc_data.csv")
    series_uid_to_reject = list(df_boundary_dsc_lung_lobe[df_boundary_dsc_lung_lobe.dice < 0.9].SeriesUID.values)
    # series_uid_to_use = list(pd.read_csv("/home/users/shubham.kumar/projects/lung_lobe_segmentation/VESSEL12_dataset.csv").SeriesUID.values)
    for i, row in df.iterrows():
        series_uid = row["SeriesUID"]
        # use_lobe_fused = False
        if series_uid in series_uid_to_reject :
            continue

        datapath = os.path.join(data_root["image"], f"{str(series_uid)}.h5")
        if os.path.exists(datapath) == True :
            if keep_lung_only :
                lower , upper = int(row["lung_lower"]) , int(row["lung_upper"])
            else:
                f2 = h5py.File(datapath, 'r')
                lower , upper = 0 , f2["image"].shape[0]
                f2.close()

            if load_scan :
                data_dict.append(
                        {   
                            "datapath" : datapath,
                            "series_uid": series_uid,
                            "phase": phase,
                            "lower": lower,
                            "upper": upper,
                            "invert_ct": invert_ct,
                            # "use_lobe_fused": use_lobe_fused
                        }
                )
            else: 
                batch = _create_batch_index(lower, upper, num_slices_per_volume, stride)
                # logger.debug(batch)
                for low, high in batch:

                    data_dict.append(
                        {   
                            "datapath" : datapath,
                            "series_uid": series_uid,
                            "phase": phase,
                            "lower": low,
                            "upper": high,
                            "invert_ct": invert_ct,
                            # "use_lobe_fused":use_lobe_fused
                        }
                    )
        else :
            logger.warning(f"hdf5 dataset {datapath} does not exist for series_uid : {series_uid}")
    logger.info(f"{len(data_dict)}")
    return data_dict