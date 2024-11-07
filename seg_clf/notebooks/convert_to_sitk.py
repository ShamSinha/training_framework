import pandas as pd
from tqdm.auto import tqdm
import os
import SimpleITK as sitk
import sqlite3
from qer_utils.db import get_mongo_db
db = get_mongo_db()
from qer.utils.imageoperations.resampler import load_raw_sitk_img
from loguru import logger
import multiprocessing


path = "/home/users/shubham.kumar/projects/ICH_classification_segmentation/qct_training_framework/all_studies.sqlite"
con = sqlite3.connect(path)
df = pd.read_sql_query("SELECT * from nlp", con)

# Verify that result of SQL query is stored in the dataframe
con.close()

df_q25k = df[(df.Status == "qure25k")]
df_q25k = df_q25k[(df_q25k["ANY"] != 100) & (df_q25k["SDH"] == 1)]


# base_path = "/data_nas5/qer/shubham/new_test_set_ICH/"
# data_already_present = os.listdir(base_path)

# study_uids_done = [i[:-len(".nii.gz")] for i in data_already_present]
# study_uids_yet_to_process = list(set(df_q25k.StudyUID.values) - set(study_uids_done))



# def save_data(study_uid):
#     try : 
#         row = df_q25k.loc[df_q25k["StudyUID"] == study_uid]
#         series_uid = row["SeriesUID"].values[0]
#         series_dict = db.dicoms.find_one({'_id' : series_uid})
#         filepaths = [x['FilePath'] for x in series_dict["InstancesList"]]
#         filepaths = [filepath.replace( "/data_nas2/raw/" , "/data_nas3/raw/hct/")  for filepath in filepaths]

#         for path in filepaths :
#             if not os.path.exists(path) :
#                 return

#         sitk_img = load_raw_sitk_img(filepaths)
#         sitk.WriteImage(sitk_img , f"/data_nas5/qer/shubham/new_test_set_ICH/{study_uid}.nii.gz")
#     except Exception as e :
#         logger.debug(e)
#         return


# with multiprocessing.Pool(8) as p:
#     list(tqdm(p.imap(save_data, study_uids_yet_to_process), total=len(study_uids_yet_to_process), desc="Caching"))

for idx in tqdm(df_q25k.index) : 
    try : 
        row = df_q25k.loc[idx]
        series_uid = row["SeriesUID"]
        study_uid = row["StudyUID"]
        if not os.path.exists(f"/data_nas5/qer/shubham/new_test_set_ICH/{study_uid}.nii.gz"):
            series_dict = db.dicoms.find_one({'_id' : series_uid})
            filepaths = [x['FilePath'] for x in series_dict["InstancesList"]]
            new_filepaths = [filepath.replace( "/data_nas2/raw/" , "/data_nas3/raw/hct/")  for filepath in filepaths]

            invalid_paths = False
            for path in new_filepaths :
                if not os.path.exists(path) :
                    invalid_paths = True
                    break

            if invalid_paths : 
                new_filepaths = [filepath.replace( "/data_nas2/processed/HeadCT/" , "/data_nas3/processed/hct/")  for filepath in filepaths]
      

            sitk_img = load_raw_sitk_img(new_filepaths)
            sitk.WriteImage(sitk_img , f"/data_nas5/qer/shubham/new_test_set_ICH/{study_uid}.nii.gz")

    except KeyboardInterrupt:
        break
    except Exception as e :
        logger.debug(e)
        continue






# df_new_test_set = pd.read_csv("/home/users/shubham.kumar/projects/qct_training_framework/notebooks/new_test_set.csv")
# df_to_work = df_new_test_set[(df_new_test_set["fusion_v5"].isnull()) & (df_new_test_set["Source"] == "sqlite_val")]

# for idx in tqdm(df_to_work.index) :
#     try :
#         filepath = df_to_work.loc[idx , "FilePath"]
#         study_uid = df_to_work.loc[idx , "StudyUID"]
#         if os.path.exists( f"/data_nas5/qer/shubham/new_test_set_ICH/{study_uid}.nii.gz") :
#             continue
#         if os.path.exists(filepath) :
#             if os.path.isdir(filepath) :
#                 sitk_img = load_raw_sitk_img(filepath)
#                 sitk.WriteImage(sitk_img , f"/data_nas5/qer/shubham/new_test_set_ICH/{study_uid}.nii.gz")
#     except KeyboardInterrupt:
#         break
#     except Exception as e:
#         continue

