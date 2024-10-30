from safetensors import safe_open
import glob
from typing import List, Dict
import pandas as pd
import os
import random
from loguru import logger
import torch

def _load_data_neg(li_df: List[pd.DataFrame] , directorys: List[str], data_source: List[str]) :
    li = []
    for j in range(len(directorys)):
        directory = directorys[j]
        df = li_df[j]
        source = data_source[j]
        logger.debug(source)
        for i in df.index :
            suid = df.loc[i, "SeriesUID"]
            bbox = eval(df.loc[i, "bbox"])
            path = os.path.join(directory , f"{suid}.safetensors")
            if os.path.exists(path) : 
                temp_data = {}
                temp_data["datapath"] = path
                temp_data["bbox"] = bbox
                temp_data["label"] = 0
                temp_data["source"] = source
                li.append(temp_data)
    return li


def _load_data_pos(li_df: List[pd.DataFrame] , directorys: List[str], data_source: List[str]) :
    li = []
    for j in range(len(directorys)):
        directory = directorys[j]
        df = li_df[j]
        source = data_source[j]
        logger.debug(source)
        for i in df.index :
            row = df.loc[i]
            suid = row["SeriesUID"]
            xc = row["xc"]
            yc = row["yc"]
            zc = row["zc"]
            w = row["w"]
            h = row["h"]
            d = row["d"]

            bbox = [max(int(zc-max(d/2 , 8)),0), int(zc+max(d/2 , 8)) , max(int(yc-max(h/2 , 24)), 0 ), int(yc+max(h/2 , 24)), max(int(xc-max(w/2 , 24)),0), int(xc+max(w/2 , 24))]
            
            path = os.path.join(directory , f"{suid}.safetensors")
            if os.path.exists(path) : 
                temp_data = {}
                temp_data["datapath"] = path
                temp_data["bbox"] = bbox
                temp_data["label"] = 1
                temp_data["source"] = source
                li.append(temp_data)
    return li
    

def fpr_sl_data_loader(
    pos_csvs: List[str],
    neg_csvs: List[str],
    pos_directorys: List[str],
    neg_directorys: List[str],
    phase: str,
    frac: float,
    data_source: List[str]
):
    """
    Returns:
    --------
    List of dicts. Each dicts have information about that datapoint.
    """
    li_df_neg = []
    dir_neg = []

    for i in range(len(neg_csvs)) :
        df = pd.read_csv(neg_csvs[i])
        df = df[df["label"]== 0]
        df = df[df["Status"] == phase]
        if len(df) > 0 : 
            req_df = []
            for j in range(5) :
                k = len(df[df["cluster"] == j])
                req_df.append(df[df["cluster"] == j].sample(n=min(3000 , k), random_state= 33))
            final_df = pd.concat(req_df)
            final_df = final_df.sample(frac=frac)
            li_df_neg.append(final_df)
            dir_neg.append(neg_directorys[i])

    neg_li_data = _load_data_neg(li_df_neg , dir_neg , data_source)

    li_df_pos = []
    dir_pos = []

    for i in range(len(pos_csvs)) :
        df = pd.read_csv(pos_csvs[i])
        df = df[df["Status"] == phase]
        if len(df) > 0 : 
            final_df = df
            final_df = final_df.sample(frac=frac)
            li_df_pos.append(final_df)
            dir_pos.append(pos_directorys[i])

    pos_li_data = _load_data_pos(li_df_pos , dir_pos, data_source)

    logger.debug(f"{phase} : positive: {len(pos_li_data)} , negative: {len(neg_li_data)}")

    all_li_data = neg_li_data + pos_li_data
    
    random.shuffle(all_li_data)

    return all_li_data


########################

def _load_data(li_df: List[pd.DataFrame] , directorys: List[str], data_source) :
    pos_label = 0
    neg_label = 0
    li = []
    for j in range(len(directorys)):
        directory = directorys[j]
        df = li_df[j]
        source = data_source[j]
        logger.debug(source)
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
                    
                    temp_data["label"] = tensor_slice[temp_data["start_idx"] : temp_data["end_idx"]].item()

                    temp_data["source"] = source

                    if temp_data["label"] == 0 :
                        neg_label += 1
                    if temp_data["label"] == 1 :
                        pos_label += 1

                    li.append(temp_data)

    logger.debug(f"positive: {pos_label} , negative: {neg_label}")

    return li
    

def videomae_data_loader(
    csvs: List[str],
    directorys: List[str],
    phase: str,
    frac: float,
    data_source: List[str]
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

    return _load_data(li_df, dir, data_source)


def fpr_sl_mae_data_loader(
    sl_inputs: Dict, 
    mae_inputs: Dict,
    phase: str,
):
    """
    Returns:
    --------
    List of dicts. Each dicts have information about that datapoint.
    """

    li_mae = videomae_data_loader(mae_inputs["csvs"], mae_inputs["directorys"], phase , mae_inputs["frac"], mae_inputs["data_source"])

    for j in range(len(li_mae)) :
        li_mae[j]["data_type"] = 1

    p = []
    for out in li_mae :
        p.append(out["source"])
    
    li_sl = fpr_sl_data_loader(sl_inputs["pos_csvs"], sl_inputs["neg_csvs"]  , sl_inputs["pos_directorys"]  , sl_inputs["neg_directorys"] , phase , sl_inputs["frac"], sl_inputs["data_source"])

    for j in range(len(li_sl)) :
        li_sl[j]["data_type"] = 0

    q = []
    for out in li_sl :
        q.append(out["source"])

    from collections import Counter

    unique_counts = Counter(p)
    logger.debug(unique_counts)

    unique_counts = Counter(q)
    logger.debug(unique_counts)

    return li_mae + li_sl