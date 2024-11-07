import os
import pickle
import pandas as pd
from tqdm.auto import tqdm
import torch
import numpy as np
import h5py
from qer_utils.db import get_mongo_db
db = get_mongo_db()
from qer.ai.predictor.get_predictions import load_and_run_model


from qure_dicom_utils.dicom.series import geometry
from skimage import morphology
import SimpleITK as sitk
from loguru import logger
import multiprocessing

df_test_sdh_vol =pd.read_csv("quant_SDH_volume_logits_correction_v4.csv")

df_quant_SDH_test = pd.read_csv("/home/users/shubham.kumar/projects/ICH_classification_segmentation/quant_SDH_q25k_outputs.csv")
df_test = pd.read_csv("/home/users/shubham.kumar/projects/ICH_classification_segmentation/ICH_dataset_test.csv")
a = df_test[df_test.StudyUID.isin(df_quant_SDH_test.StudyUID.values)].StudyUID.values
df_test_SDH = df_quant_SDH_test[df_quant_SDH_test["StudyUID"].isin(a)]
df_test_SDH = df_test_SDH[~df_test_SDH["StudyUID"].isin(df_test_sdh_vol.StudyUID.values)]
sdh_study_uids = list(df_test_SDH.StudyUID.values)


def get_headct_arr(study_uid) : 
    base_path = '/cache/fast_data_nas8/qer/shubham/ich'
    path = os.path.join(base_path, study_uid)
    h5_data = path + '.h5'
    
    if os.path.exists(h5_data) :
        f2 = h5py.File(h5_data, 'r')
        image = f2['image']
        arr = np.array(image)
        return arr
    else:
        return None

df_spacing= pd.read_csv("spacing_sdh_test_set.csv")

def get_spacing(study_uid): 
    rows = df_spacing.loc[df_spacing.StudyUID == study_uid]
    return (rows['delta_x'].values[0], rows['delta_y'].values[0], rows['delta_z'].values[0])
    
def _post_process(mask, img_arr):
    mask = morphology.dilation(mask, morphology.ball(2))
    mask = np.array(
        [
            morphology.dilation(mask[i], morphology.disk(3))
            for i in range(mask.shape[0])
        ]
    )

    mask = mask * img_arr
    mask[mask < 40] = 0
    mask[mask > 100] = 0
    mask[mask > 0] = 1

    return mask
    

# ths = [x / 20.0 for x in range(1, 20, 1)]
ths = [0.2,0.25,0.3,0.35,0.4,0.45,0.55,0.6]
out_test_sdh = []
for study_uid in tqdm(sdh_study_uids) : 
    try : 
        if os.path.exists(f'/cache/fast_data_nas8/qer/shubham/ich_pickle/{study_uid}.pkl') :
            with open(f'/cache/fast_data_nas8/qer/shubham/ich_pickle/{study_uid}.pkl', 'rb') as file:


                # Call load method to deserialze
                myvar = pickle.load(file)
                quant_output = myvar.copy()
                img_arr = get_headct_arr(study_uid)
                if img_arr is None :
                    continue
                else :
                    img = sitk.GetImageFromArray(img_arr)
                a,b,_ = get_spacing(study_uid)
                # quant_output = load_and_run_model("hemorrhages_quantification",img,debug_mode = True)
                acute_mask_logits_arr = sitk.GetArrayFromImage(quant_output['results']['heatmaps']['acute_mask_logits'])
                volume_old = quant_output['results']['quantification value']['acute_pp']
                # logger.debug(acute_mask_logits_arr.shape)
                activation = torch.nn.Softmax(dim =1)

                logits_acute_final = torch.stack([-torch.Tensor(acute_mask_logits_arr) , torch.Tensor(acute_mask_logits_arr)],dim =1)
                c = activation(torch.Tensor(logits_acute_final)).numpy()
                pred = c[:,1,:,:]
#                 gt = get_gt_arr(study_uid)
#                 logger.debug(gt.shape)
                # logger.debug(pred.shape)
                
                img_arr = get_headct_arr(study_uid)
                a,b,_ = get_spacing(study_uid)
    
                for th in ths :
                    mask = pred > th
                    mask = _post_process(mask, img_arr)
#                 dice_score = dice_coefficient(gt, mask)
                
                    out_test_sdh.append({"StudyUID":study_uid , f"volume_new_{th}":(np.sum(mask)*a*b*5)/1000 , "volume_old":volume_old})

    except KeyboardInterrupt:
        break
    except Exception as e:
        logger.debug(e)
        continue


def get_volume(study_uid):
    try : 
        if os.path.exists(f'/cache/fast_data_nas8/qer/shubham/ich_pickle/{study_uid}.pkl') :
            with open(f'/cache/fast_data_nas8/qer/shubham/ich_pickle/{study_uid}.pkl', 'rb') as file:

                # Call load method to deserialze
                myvar = pickle.load(file)
                quant_output = myvar.copy()
                acute_mask_logits_arr = sitk.GetArrayFromImage(quant_output['results']['heatmaps']['acute_mask_logits'])
                volume_old = quant_output['results']['quantification value']['acute_pp']
                # logger.debug(acute_mask_logits_arr.shape)
                activation = torch.nn.Softmax(dim =1)

                logits_acute_final = torch.stack([-torch.Tensor(acute_mask_logits_arr) , torch.Tensor(acute_mask_logits_arr)],dim =1)
                c = activation(torch.Tensor(logits_acute_final)).numpy()
                pred = c[:,1,:,:]
#                 gt = get_gt_arr(study_uid)
#                 logger.debug(gt.shape)
                # logger.debug(pred.shape)
                
                img_arr = get_headct_arr(study_uid)
                a,b,_ = get_spacing(study_uid)
    
                for th in ths :
                    mask = pred > th
                    mask = _post_process(mask, img_arr)
#                 dice_score = dice_coefficient(gt, mask)
                
                    out_test_sdh.append({"StudyUID":study_uid , f"volume_new_{th}":(np.sum(mask)*a*b*5)/1000 , "volume_old":volume_old})

    except KeyboardInterrupt:
        return 
    except Exception as e:
        logger.debug(e)
        return

# with multiprocessing.Pool(8) as p:
#     list(tqdm(p.imap(get_volume, sdh_study_uids), total=len(sdh_study_uids), desc="Caching"))


df = pd.DataFrame.from_records(out_test_sdh) 
print(len(df))

pd.concat([df_test_sdh_vol , df]).to_csv("quant_SDH_volume_logits_correction_v4.csv",index =False)
            