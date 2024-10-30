import os
import math
import pandas as pd
from loguru import logger
from typing import Dict, List, Optional, Union
from tqdm.auto import tqdm
import SimpleITK as sitk
import numpy as np
import h5py

def _create_batch(file_names: List[str], files_per_batch: int , stride: int):
    batch = []
    if len(file_names) == files_per_batch :
        batch.append(file_names)
    else :
        num_batch = math.floor(len(file_names)/(files_per_batch - stride))
        start_idx = 0
        end_idx = files_per_batch 
        batch.append(file_names[start_idx : end_idx])
        for i in range(1, num_batch) :
            start_idx = start_idx + files_per_batch - stride
            end_idx = end_idx + files_per_batch - stride
            batch.append(file_names[start_idx:end_idx])

        if len(batch[-1]) != files_per_batch : 
            shift = files_per_batch - len(batch[-1]) 
            batch[-1] = file_names[start_idx-shift:end_idx]
    return batch

def _create_batch_index(lower_idx, upper_idx , slice_per_batch: int , stride: int) :
    batch = []
    if upper_idx - lower_idx == slice_per_batch :
        batch.append((lower_idx, upper_idx))
    else :
        num_batch = math.floor((upper_idx - lower_idx)/(slice_per_batch - stride))
        start_idx = lower_idx 
        end_idx = lower_idx + slice_per_batch
        batch.append((start_idx, end_idx))
        for i in range(1, num_batch) :
            start_idx = start_idx + slice_per_batch - stride
            end_idx = end_idx + slice_per_batch - stride
            if end_idx > upper_idx :
                start_idx = upper_idx - slice_per_batch
                end_idx = upper_idx
            else: 
                batch.append((start_idx, end_idx))
    return batch

def _load_seg_dataset_ctloader(
    df: pd.DataFrame, data_root: Dict , num_slices_per_volume: int , stride: int , phase: str, keep_lung_only: bool ,invert_ct: bool, load_scan: bool ):
    
    # df_boundary_dsc_lung_lobe = pd.read_csv("/home/users/shubham.kumar/projects/lung_lobe_segmentation/boundary_lung_lobe_dsc_data.csv")
    # series_uid_to_reject = df_boundary_dsc_lung_lobe[df_boundary_dsc_lung_lobe.dice < 0.9].SeriesUID.values
    data_dict = []
    for i, row in df.iterrows():
        series_uid = row["SeriesUID"]
        # if series_uid in series_uid_to_reject :
        #     continue
        
        filepath = os.path.join(data_root["image"], str(series_uid))
        lungpath = os.path.join(data_root["lung_annot"], str(series_uid))
        lobepath = os.path.join(data_root["lobe_annot"], str(series_uid))
        num_slices = len(os.listdir(filepath))
        filenames = []
        lungfilenames = []
        lobefilenames = []
        if keep_lung_only :
            lower , upper = int(row["lung_lower"]) , int(row["lung_upper"])
        else:
            lower , upper = 0 , num_slices
   
        for slice_ in range(lower, upper):
            slice_num = (3 - len(str(slice_))) * "0" + str(slice_)
            data_path = os.path.join(filepath, f"image_{slice_num}.dcm")
            lung_annot_path = os.path.join(lungpath, f"image_{slice_num}.dcm")
            lobe_annot_path = os.path.join(lobepath, f"image_{slice_num}.dcm")
            filenames.append(data_path)
            lungfilenames.append(lung_annot_path)
            lobefilenames.append(lobe_annot_path)   

        if invert_ct :
            filenames = filenames[::-1]
            lungfilenames = lungfilenames[::-1]
            lobefilenames = lobefilenames[::-1] 

        if load_scan : 
            if len(filenames) == len(lungfilenames) == len(lobefilenames) :
                data_dict.append(
                    {   
                        "image" : filenames,
                        "lung_label": lungfilenames,
                        "lobe_label": lobefilenames,
                        "series_uid": series_uid,
                        "phase": phase,
                    }
                )
            else :
                logger.warning(f"image: {len(filenames)} , lung_label: {len(lungfilenames)}  lobe_label: {len(lobefilenames)} size are different for series_uid : {series_uid}")
        else:
            batch = _create_batch(filenames, num_slices_per_volume, stride)
            lungbatch = _create_batch(lungfilenames, num_slices_per_volume, stride)
            lobebatch = _create_batch(lobefilenames, num_slices_per_volume, stride)

            if len(batch) == len(lungbatch) == len(lobebatch) :
                for i in range(len(batch)) : 
                    data_dict.append(
                        {   
                            "image" : batch[i],
                            "lung_label": lungbatch[i],
                            "lobe_label": lobebatch[i],
                            "series_uid": series_uid,
                            "phase": phase,
                        }
                    )
            else :
                logger.warning(f"image: {len(batch)} , lung_label: {len(lungbatch)}  lobe_label: {len(lobebatch)} size are different for series_uid : {series_uid}")

    return data_dict

def _load_whole_seg_dataset_ctloader(
    df: pd.DataFrame, data_root: Dict , phase: str, keep_lung_only: bool, invert_ct: bool, ext: str ):
    data_dict = []
    for i, row in df.iterrows():
        series_uid = row["SeriesUID"]
        
        filepath = os.path.join(data_root["image"], f"{str(series_uid)}{ext}")
        lungpath = os.path.join(data_root["lung_annot"],f"{str(series_uid)}.nii.gz")
        lobepath = os.path.join(data_root["lobe_annot"], f"{str(series_uid)}.nii.gz")

        if os.path.exists(filepath) == os.path.exists(lungpath) == os.path.exists(lobepath) == True :
            if keep_lung_only :
                lower , upper = int(row["lung_lower"]) , int(row["lung_upper"])
            else:
                lower , upper = 0 , None
        
            data_dict.append(
                        {   
                            "image" : filepath,
                            "lung_label": lungpath,
                            "lobe_label": lobepath,
                            "series_uid": series_uid,
                            "phase": phase,
                            "lower": lower,
                            "upper": upper,
                            "invert_ct": invert_ct
                        }
                    )
            
        else :
            logger.warning(f"image: {filepath} , lung_label: {lungpath}  lobe_label: {lobepath} does not exist for series_uid : {series_uid}")
    return data_dict

def _load_seg_whole_scan_resampled_dataset_ctloader(
    df: pd.DataFrame, data_root: Dict , num_slices_per_volume: int , phase: str):

    data_dict = []
    for i, row in df.iterrows():
        series_uid = row["SeriesUID"]
        
        filepath = os.path.join(data_root["image"], str(series_uid))
        lungpath = os.path.join(data_root["lung_annot"], str(series_uid))
        lobepath = os.path.join(data_root["lobe_annot"], str(series_uid))
        filenames = []
        lungfilenames = []
        lobefilenames = []
        for slice_ in range(int(row["lung_lower"]) , int(row["lung_upper"])):
            slice_num = (3 - len(str(slice_))) * "0" + str(slice_)
            data_path = os.path.join(filepath, f"image_{slice_num}.dcm")
            lung_annot_path = os.path.join(lungpath, f"image_{slice_num}.dcm")
            lobe_annot_path = os.path.join(lobepath, f"image_{slice_num}.dcm")
            filenames.append(data_path)
            lungfilenames.append(lung_annot_path)
            lobefilenames.append(lobe_annot_path)   

        if len(filenames) == len(lungfilenames) == len(lobefilenames) :
            num_cubes = math.ceil(len(lungfilenames)/ num_slices_per_volume)
            for i in range(num_cubes) :
                data_dict.append(
                        {   
                            "image" : filenames[i::num_cubes],
                            "lung_label": lungfilenames[i::num_cubes],
                            "lobe_label": lungfilenames[i::num_cubes],
                            "series_uid": series_uid,
                            "phase": phase,
                        }
                    )
        else :
            logger.warning(f"image: {len(filenames)} , lung_label: {len(lungfilenames)}  lobe_label: {len(lobefilenames)} size are different for series_uid : {series_uid}")
    return data_dict

def lung_segmentation_data_loader(
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
    # df_boundary_dsc_lung_lobe = pd.read_csv("/home/users/shubham.kumar/projects/lung_lobe_segmentation/boundary_lung_lobe_dsc_data.csv")
    # series_uid_to_reject = list(df_boundary_dsc_lung_lobe[df_boundary_dsc_lung_lobe.dice < 0.9].SeriesUID.values)
    # series_uid_to_use = list(pd.read_csv("/home/users/shubham.kumar/projects/lung_lobe_segmentation/VESSEL12_dataset.csv").SeriesUID.values)
    for i, row in df.iterrows():
        series_uid = row["SeriesUID"]
        # use_lobe_fused = False
        # if series_uid in series_uid_to_reject :
        #     continue

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