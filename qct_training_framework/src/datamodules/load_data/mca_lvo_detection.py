import os
import math
import pandas as pd
from loguru import logger
from typing import Dict, List, Optional, Union
from tqdm.auto import tqdm
import SimpleITK as sitk
import numpy as np
import h5py



def mca_lvo_data_loader(
    csvs: List[str],
    data_root: List[Dict],
    sample_frac: Optional[Union[float, List[float]]],
    sample_size: Optional[Union[int, List[int]]],
    random_state: int,
    phase: str,

):
    """
    Base extraction function for Lung and Lobe segmentation.
    Args:
    -----
        csvs: List of csvs containing the data.
        data_root: Root directory of the data.
    """

    li = []

    for i in range(len(csvs)):
        csv = csvs[i]
        df = pd.read_csv(csv)
        df = df[df.Status == phase]
        frac = sample_frac[i]
        # logger.debug(f"Loading {phase} data from {csv} with frac {frac} and size {sample_size}.")
        df = df.sample(n = sample_size , frac = frac , axis = 0 , random_state = random_state)
        li.extend(_load_hdf5_dataset_ctloader(df, data_root[i], phase))
    
    return li

def _load_hdf5_dataset_ctloader(
    df: pd.DataFrame, data_root: Dict , phase: str):
    data_dict = []

    for i, row in df.iterrows():
        series_uid = row["SeriesUID"]
        tilt_corr = row["tilt_corr"]
        if tilt_corr : 
            datapath = os.path.join(data_root["tilt_corr"], f"{str(series_uid)}.h5")
        else :
            datapath = os.path.join(data_root["raw"], f"{str(series_uid)}.h5")
        if os.path.exists(datapath) == True :
            data_dict.append(
                    {   
                        "datapath" : datapath,
                        "series_uid": series_uid,
                        "phase": phase,
                    }
            )
        else :
            logger.warning(f"hdf5 dataset {datapath} does not exist for series_uid : {series_uid}")
    logger.info(f"{len(data_dict)}")
    return data_dict