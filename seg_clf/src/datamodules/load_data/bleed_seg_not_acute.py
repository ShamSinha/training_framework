from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import os
import os
import ast
from tqdm.auto import tqdm
import h5py
from loguru import logger
import random

root_dir = "/data_nas3/processed/hct/Cache/all_imgs_cache.dcm/"

df_hemorrhage_mask = pd.read_csv("/home/users/shubham.kumar/projects/ICH_classification_segmentation/final_train_val_bleed_pos_seg_dataset.csv")
df_hemorrhage_mask = df_hemorrhage_mask[df_hemorrhage_mask.POSTOP==0]
df_hemorrhage_mask = df_hemorrhage_mask[df_hemorrhage_mask.Intensity=="acute"]
study_uids_having_bleed_mask = list(set(df_hemorrhage_mask.StudyUID.values))

df_bleed_seg_bbox = pd.read_csv("/home/users/shubham.kumar/projects/ICH_classification_segmentation/bleed_seg_bbox.csv")
df_normal = pd.read_csv("/home/users/shubham.kumar/projects/qct_training_framework/notebooks/train_normal_ncct.csv")
normal_study_uids = list(df_normal.sample(5000,random_state=85).StudyUID.values)

def _load_cls_data(df: pd.DataFrame, phase: str):
    data_dict = []

    for ind, row in tqdm(df.iterrows()):
        temp_dict = {}
        if phase == "test":
            base_path = "/cache/fast_data_nas8/qer/shubham/ich_seg"
            path = os.path.join(base_path, row["StudyUID"])
            temp_dict["datapath"] = path + ".h5"
        else:
            if row["ANY"] == 100:
                continue
            temp_dict["scan_label"] = row["ANY"]
            temp_dict["study_uid"] = row["StudyUID"]
            temp_dict["IPH"] = row["ICH"] or row["CONT"]
            temp_dict["SDH"] = row["SDH"]
            temp_dict["SAH"] = row["SAH"]
            temp_dict["EDH"] = row["EDH"]
            temp_dict["IVH"] = row["IVH"]
            base_path = "/cache/fast_data_nas8/qer/shubham/ich"
            path = os.path.join(base_path, row["StudyUID"])
            temp_dict["datapath"] = path + ".h5"
            if temp_dict["scan_label"] == 1 and temp_dict["study_uid"] in study_uids_having_bleed_mask: 
                try : 
                    f2 = h5py.File(temp_dict["datapath"], "r")
                    if "mask" not in f2.keys():
                        base_path = "/cache/fast_data_nas8/qer/shubham/ich_seg_train_val"
                        path = os.path.join(base_path, row["StudyUID"])
                        temp_dict["datapath"] = path + ".h5"

                        # row = df_bleed_seg_bbox.loc[df_bleed_seg_bbox["StudyUID"]== temp_dict["study_uid"]]
                        # if len(row) > 0:
                        #     temp_dict["lower"] = int(row.z1.values[0])
                        #     temp_dict["upper"] = int(row.z2.values[0])

                    f2.close()
                except:
                    logger.debug(temp_dict["datapath"])
                    continue

            if row["StudyUID"] in normal_study_uids :
                base_path = "/cache/fast_data_nas8/qer/shubham/new_test"
                valid_paths = []
                study_uid =  row["StudyUID"]
                for perp_dist in range(0,30) : 
                    path = os.path.join(base_path, f"{study_uid}_{perp_dist}.h5" )
                    if os.path.exists(path) :
                        valid_paths.append(path)
                if len(valid_paths) > 0 : 
                    temp_dict["datapath"] = random.choice(valid_paths)
                    temp_dict["SDH"] = row["SDH"]
                    temp_dict["scan_label"] = 1
                else :
                    continue
            try:
                temp_dict["crop"] = np.array(ast.literal_eval(row["crop"]))
            except:
                temp_dict["crop"] = None

        if os.path.exists(temp_dict["datapath"]):
            if temp_dict["datapath"] in ["/cache/fast_data_nas8/qer/shubham/ich/1.2.840.113619.2.55.3.2831165742.939.1462502069.683.h5",
                                         "/cache/fast_data_nas8/qer/shubham/ich/1.2.840.113619.2.81.290.1.930.20140413.223950.h5",
                                         "/cache/fast_data_nas8/qer/shubham/test/1.3.6.1.4.1.25403.52234692458.1372.20150116103719.1_7.h5"]:
                continue
            data_dict.append(temp_dict)
    return data_dict


def ich_seg_data_loader_csv(
    csv_path: List[str],
    phase: str,
    num_samples_per_class: Optional[int] = None,
    frac_per_class: Optional[Dict[str, float]] = None,
    sample_frac: Optional[float] = None,
):
    """
    Data  extractor function for old meta classifier csv format.
    Args:
    -----
    csvs: list of meta classifier csv paths
    phase: training phase
    cls2idx_dict: class name to id mappings
    characteritic_type: characteristics we are considering
    srcs_key: column name which have the dataset source information
    data_srcs: which all sources to consider.
    Returns:
    --------
    List of dicts. Each dicts have information about that datapoint.
    """

    df = pd.concat([pd.read_csv(p) for p in csv_path])
    df = df.sample(frac=sample_frac, random_state=9)

    if phase == "val":
        df = df[df["StudyUID"].isin(study_uids_having_bleed_mask)]
        df = df.sample(n=100)
    if phase == "train" :
        df = df[df["StudyUID"].isin(study_uids_having_bleed_mask+ normal_study_uids)]
        df = df.sample(n=100)

    return _load_cls_data(df, phase)



