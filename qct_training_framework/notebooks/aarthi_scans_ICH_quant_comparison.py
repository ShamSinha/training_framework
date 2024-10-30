import os
import numpy as np
import pandas as pd

from scipy import stats as st
from functools import reduce
from tqdm.auto import tqdm
from qer.ai.predictor.get_predictions import load_and_run_model
from qer.utils.imageoperations.resampler import load_raw_sitk_img
from skimage import morphology
from qer.ai.utils.postprocessing import get_sitk_mask, mask_volume
import SimpleITK as sitk

df = pd.read_csv("/data_nas5/qer/shubham/aarthi_scans_ICH_quant_test/AarthiScans_Jan_Oct.csv")


uids  = df.SeriesInstanceUID.values


def post_process(mask, sitk_img):
    img_arr = sitk.GetArrayFromImage(sitk_img)
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

li = []
for uid in tqdm(uids) :
    filepath = f"/data_nas5/qer/shubham/aarthi_scans_ICH_quant_test/data/{uid}"
    if os.path.exists(filepath) : 
        try : 
            img = load_raw_sitk_img(filepath)
            # v3_output = load_and_run_model(model_name_or_config= "hemorrhages_v3" , raw_sitk_img=img)
            # V3_score = v3_output["results"]["scores"]["ICH"]
            
            # quant_output = load_and_run_model(model_name_or_config= "hemorrhages_quantification" , raw_sitk_img=img ,debug_mode=True)
            
            # acute_logits = quant_output["results"]["heatmaps"]["acute_logits"]
            # threshold = 0.3
            # acute_old_mask = sitk.GetArrayFromImage(acute_logits)
            # acute_old_mask[acute_old_mask > threshold] = 1
            # acute_old_mask[acute_old_mask <= threshold] = 0
                
            # acute_mask_pp = post_process(acute_old_mask, img)
            # acute_mask_pp_sitk = get_sitk_mask(acute_mask_pp, img)
            # acute_vol_pp_old = mask_volume(acute_mask_pp_sitk)
            
            # acute_pp_vol_new = quant_output["results"]["quantification value"]["acute_pp"]

            # li.append({"SeriesUID" : uid , "v3_score":V3_score , "volume_new" : acute_pp_vol_new , "volume_old" : acute_vol_pp_old})

            vjun23_output = load_and_run_model(model_name_or_config= "hemorrhages_vjun23" , raw_sitk_img=img)
            vjun23_score = vjun23_output["results"]["scores"]["ICH"]

            hemorrhages_output = load_and_run_model(model_name_or_config= "hemorrhages" , raw_sitk_img=img)
            hemorrhages_score = hemorrhages_output["results"]["scores"]["ICH"]

            li.append({"SeriesUID" : uid , "hemorrhages_score": hemorrhages_score ,  "vjun23_score":vjun23_score })

            print(li[-1])
        except KeyboardInterrupt :
            break
        except Exception as e :
            print(e)
            continue

pd.DataFrame.from_records(li).to_csv("aarthi_scans_ICH_scores.csv" , index= False)

