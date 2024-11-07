from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import os
import os
import ast
import sqlite3
from loguru import logger
from pydicom import dcmread
import math
from tqdm.auto import tqdm

root_dir = "/data_nas3/processed/hct/Cache/all_imgs_cache.dcm/"

path = "/home/users/shubham.kumar/projects/ICH_classification_segmentation/qct_training_framework/hemorrhage_annotations.sqlite"
con = sqlite3.connect(path)
df_hemorrhage_mask = pd.read_sql_query("SELECT * from hemorrhage_annotations", con)
con.close()


def get_mask_arr_path(study_uid):
    rows = df_hemorrhage_mask[df_hemorrhage_mask.StudyUID == study_uid]
    if len(rows) > 0:
        study_uid = rows.iloc[0].StudyUID
        mask_path = rows.iloc[0].MaskPath[:-7]
        filepath = os.path.join(root_dir, mask_path)
        filenames = []
        for slice_ in range(len(os.listdir(filepath))):
            slice_num = (3 - len(str(slice_))) * "0" + str(slice_)
            data_path = os.path.join(filepath, f"image_{slice_num}.dcm")
            filenames.append(data_path)
        return filenames
    return -100


df_slice_gt = pd.read_csv(
    "/home/users/shubham.kumar/projects/ICH_classification_segmentation/ICH_dataset_slice_gts.csv"
)
slice_gt_study_uids = df_slice_gt["StudyUID"].values

def _load_cls_data(df: pd.DataFrame, phase: str):
    data_dict = []

    for ind, row in tqdm(df.iterrows()):
        temp_dict = {}
        base_path = "/cache/fast_data_nas8/qer/sujith"
        if phase == "test":
            path = os.path.join(base_path, row["study_uid"])
        else:
            path = os.path.join(base_path, row["StudyUID"])

        temp_dict["datapath"] = path + ".h5"

        if os.path.exists(temp_dict["datapath"]):
            if phase != "test":
                if row["ANY"] == 100:
                    continue
                temp_dict["ICH"] = row["ANY"]
                temp_dict["scan_label"] = row["ANY"]
                temp_dict["study_uid"] = row["StudyUID"]
                temp_dict["mask"] = get_mask_arr_path(row["StudyUID"])
                temp_dict["IPH"] = row["ICH"] or row["CONT"]
                temp_dict["SDH"] = row["SDH"]
                temp_dict["SAH"] = row["SAH"]
                temp_dict["EDH"] = row["EDH"]
                temp_dict["IVH"] = row["IVH"]
            else:
                temp_dict["scan_label"] = row["ICH"]
                temp_dict["mask"] = get_mask_arr_path(row["study_uid"])
                temp_dict["ICH"] = row["ICH"]
                temp_dict["study_uid"] = row["study_uid"]  # useful for testing
                temp_dict["IPH"] = row["IPH"]
                temp_dict["SDH"] = row["SDH"]
                temp_dict["SAH"] = row["SAH"]
                temp_dict["EDH"] = row["EDH"]
                temp_dict["IVH"] = row["IVH"]

            if phase != "test":
                if row["StudyUID"] in slice_gt_study_uids:
                    slice_gt = df_slice_gt.loc[
                        df_slice_gt["StudyUID"] == row["StudyUID"], "slice_gt"
                    ].values[0]
                    temp_dict["slice_label"] = ast.literal_eval(slice_gt)
                else:
                    if temp_dict["ICH"] == 1:
                        temp_dict["slice_label"] = [-100] * 32
                    else:
                        temp_dict["slice_label"] = [0] * 32
        try:
            temp_dict["crop"] = np.array(ast.literal_eval(row["crop"]))
        except:
            temp_dict["crop"] = None
        data_dict.append(temp_dict)

    return data_dict


def ich_cls_data_loader_csv(
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

    # if phase == "test":
    #     # df = df[(df.SDH == 1)]
    #     df_neg = pd.read_csv(
    #         "/home/users/shubham.kumar/projects/ICH_classification_segmentation/test_negative_ICH_sampled.csv"
    #     )
    #     df_pos = pd.read_csv("/home/users/shubham.kumar/projects/qct_training_framework/notebooks/fusion_v5_crop.csv")

    #     done_study_uids = list(df_neg.study_uid.values) + list(df_pos.StudyUID.values)

    #     df = df[~df["study_uid"].isin(done_study_uids)]

    if num_samples_per_class is not None :
        if phase != "test" :
            df["IPH"] = df["ICH"] | df["CONT"]
            df_pos = df[(df['ANY'] == 1)]
            pos_li = []
            for label in frac_per_class.keys():
                pos_li.append(df_pos[df_pos[label] == 1].sample(math.ceil(num_samples_per_class*frac_per_class[label]) , random_state=2))
            df_pos_final = pd.concat(pos_li)
            df_pos_final.drop_duplicates(inplace=True)

            df_neg = df[(df['ANY'] == 0)].sample(num_samples_per_class, random_state=2)

            df_final = pd.concat([df_pos_final , df_neg])

            logger.debug(len(df_final))

            return _load_cls_data(df_final, phase)

    return _load_cls_data(df, phase)
