from safetensors import safe_open
import glob
from typing import List
import pandas as pd
import os
from loguru import logger

def _load_data(li_df: List[pd.DataFrame] , directorys: List[str]) :
    pos_label = 0
    neg_label = 0
    li = []
    for j in range(len(directorys)):
        directory = directorys[j]
        df = li_df[j]
        for i in df.index :
            suid = df.loc[i, "SeriesUID"]
            path = os.path.join(directory , f"{suid}.safetensors")
            if os.path.exists(path) : 
                temp_data = {}
                with safe_open(path, framework="pt") as f:
                    tensor_slice = f.get_slice("labels")
                    num_videos = tensor_slice.get_shape()[0]
                for j in range(num_videos) :
                    
                    temp_data["datapath"] = path
                    temp_data["start_idx"] = j
                    if j == num_videos - 1 :
                        temp_data["end_idx"] = None
                    else :
                        temp_data["end_idx"] = j + 1
                    
                    temp_data["label"] = tensor_slice[temp_data["start_idx"] : temp_data["end_idx"]]

                    if temp_data["label"] == 0 :
                        neg_label += 1
                    if temp_data["label"] == 1 :
                        pos_label += 1
            
                    li.append(temp_data)

    logger.debug(f"pos_label: {pos_label}, neg_label:{neg_label}")

    return li
    


def videomae_data_loader(
    csvs: List[str],
    directorys: List[str],
    phase: str,
    frac: float,
):
    """
    Returns:
    --------
    List of dicts. Each dicts have information about that datapoint.
    """
    li_df = []
    dir = []

    for i in range(len(csvs)) :
        df = pd.read_csv(csvs[i])
        df = df[df["Status"] == phase]
        if len(df) > 0 : 
            df = df.sample(frac=frac)
            li_df.append(df)
            dir.append(directorys[i])

    return _load_data(li_df , dir)