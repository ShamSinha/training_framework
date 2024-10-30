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
import SimpleITK as sitk
from qer.ai.predictor.get_predictions import load_and_run_model 


root_dir = "/data_nas3/processed/hct/Cache/all_imgs_cache.dcm/"

df_hemorrhage_mask = pd.read_csv("/home/users/shubham.kumar/projects/ICH_classification_segmentation/final_train_val_bleed_pos_seg_dataset.csv")
df_hemorrhage_mask = df_hemorrhage_mask[df_hemorrhage_mask.POSTOP==0]
df_hemorrhage_mask = df_hemorrhage_mask[df_hemorrhage_mask.Intensity=="acute"]
study_uids_having_bleed_mask = list(set(df_hemorrhage_mask.StudyUID.values))

df_bleed_seg_bbox = pd.read_csv("/home/users/shubham.kumar/projects/ICH_classification_segmentation/bleed_seg_bbox.csv")
df_normal = pd.read_csv("/home/users/shubham.kumar/projects/qct_training_framework/notebooks/train_normal_ncct.csv")
normal_study_uids = list(df_normal.sample(5000,random_state=85).StudyUID.values)


df_train = pd.read_csv("/home/users/shubham.kumar/projects/ICH_classification_segmentation/ICH_dataset_train.csv")
df_train = df_train[df_train["StudyUID"].isin(study_uids_having_bleed_mask)]

def get_mls_icv(sitk_img):
    output = load_and_run_model("mls_helper_icv_quant",sitk_img)
    icv_mask = output['results']['heatmaps']['ICV']
    icv_mask_arr = sitk.GetArrayFromImage(icv_mask)
    # midline_output = load_and_run_model("mls_quantification" ,sitk_img)
    # mls_arr = sitk.GetArrayFromImage(midline_output['results']['heatmaps']['MLS'])
    # return mls_arr 
    return icv_mask_arr

def _load_cls_data(phase: str):
    li_kaggle = os.listdir("/cache/fast_data_nas8/qer/shubham/kaggle_ICH_dataset/hdf5_cache")
    kaggle_study_uids = [x[:-3] for x in li_kaggle]
    data_dict = []
    if phase == "train" :
        for study_uid in kaggle_study_uids :
            temp_dict = {}
            base_path = "/cache/fast_data_nas8/qer/shubham/kaggle_ICH_dataset/hdf5_cache/"
            path = os.path.join(base_path, f"{study_uid}.h5" )
            if os.path.exists(path) :
                f2 = h5py.File(path, "r")
                if "image" in f2.keys() and "mask" in f2.keys() :
                    temp_dict["datapath"] = path
                    temp_dict["crop"] = None
                    temp_dict["scan_label"] = 1
                    temp_dict["study_uid"] = study_uid
                    temp_dict["IPH"] = 0
                    temp_dict["SDH"] = 0
                    temp_dict["SAH"] = 0
                    temp_dict["EDH"] = 0
                    temp_dict["IVH"] = 0
                    data_dict.append(temp_dict)
                f2.close()

    return data_dict


data_dict = _load_cls_data( "train")

for k in tqdm(range(len(data_dict))) :
    try :
        data = data_dict[k]
        datapath = data["datapath"]
        f2 = h5py.File(datapath, "r+")
        if "hemi_seperator_mask" in f2.keys() : 
            continue
        arr = np.array(f2["image"])
        logger.debug(f2.keys())

        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1,1,5))
        icv_arr = get_mls_icv(img)

        # slices_to_pick = np.unique(np.argwhere(mls_arr ==1)[:,0])
        # random.shuffle(slices_to_pick)
        # slice_to_pick = slices_to_pick[0]
        # req_mls_arr = mls_arr[slice_to_pick]

        # ml_coordinates = np.argwhere(req_mls_arr==1)
        # (x1,y1) = ml_coordinates[0]
        # (x2,y2) = ml_coordinates[-1]

        # # Calculate the slope (m) and y-intercept (b)
        # m = (y2 - y1) / (x2 - x1)
        # b = y1 - m * x1

        # # Create an empty mask
        # seperator_mask = np.zeros(req_mls_arr.shape, dtype=np.uint8)

        # for x in range(req_mls_arr.shape[0]):
        #     for y in range(req_mls_arr.shape[1]):
        #         if y >= m * x + b:
        #             seperator_mask[x, y] = 2
        #         else :
        #             seperator_mask[x, y] = 1

        dseta = f2.create_dataset("icv_mask", data= icv_arr)
        # dseta = f2.create_dataset("hemi_seperator_mask", data= seperator_mask)

        f2.close()
    except KeyboardInterrupt:
        break
    except Exception as e :
        logger.debug(e)