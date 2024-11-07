from safetensors import safe_open
import glob
from typing import List
import pandas as pd
import os
import random
from loguru import logger

def _load_data_neg(li_df: List[pd.DataFrame] , directorys: List[str]) :
    li = []
    for j in range(len(directorys)):
        directory = directorys[j]
        df = li_df[j]
        for i in df.index :
            suid = df.loc[i, "SeriesUID"]
            bbox = eval(df.loc[i, "bbox"])
            path = os.path.join(directory , f"{suid}.safetensors")
            if os.path.exists(path) : 
                temp_data = {}
                temp_data["datapath"] = path
                temp_data["bbox"] = bbox
                temp_data["label"] = 0
                li.append(temp_data)
    return li


def _load_data_pos(li_df: List[pd.DataFrame] , directorys: List[str]) :
    li = []
    for j in range(len(directorys)):
        directory = directorys[j]
        df = li_df[j]
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
                li.append(temp_data)
    return li
    

def fpr_sl_data_loader(
    pos_csvs: List[str],
    neg_csvs: List[str],
    pos_directorys: List[str],
    neg_directorys: List[str],
    phase: str,
    frac: float,
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
                req_df.append(df[df["cluster"] == j].sample(n=min(100000 , k), random_state= 33))
            final_df = pd.concat(req_df)
            final_df = final_df.sample(frac=frac)
            li_df_neg.append(final_df)
            dir_neg.append(neg_directorys[i])

    neg_li_data = _load_data_neg(li_df_neg , dir_neg)

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

    pos_li_data = _load_data_pos(li_df_pos , dir_pos)

    logger.debug(f"{phase} : positive: {len(pos_li_data)} , negative: {len(neg_li_data)}")

    all_li_data = neg_li_data + pos_li_data
    
    random.shuffle(all_li_data)

    return all_li_data