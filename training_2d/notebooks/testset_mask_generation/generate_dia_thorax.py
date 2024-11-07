import torch
torch._C._jit_set_bailout_depth(0)
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm as pbar



##Load all models

thorax_model = torch.jit.load("/fast_data_e2e11/piyush/segmentation_model/thorax_segmenter_cuda.ts", map_location='cuda:0')
thorax_model.eval()
diaphragm_model = torch.jit.load("/fast_data_e2e11/piyush/segmentation_model/diaphragm_segmenter_cuda.ts", map_location='cuda:0')
diaphragm_model.eval()
# heart_model = torch.jit.load("/fast_data_e2e11/piyush/segmentation_model/v4_cardiomegaly_cuda.ts", map_location='cuda:0')
# heart_model.eval()

# img_root = "/fast_data_e2e_1/cxr/qxr_ln_data/fake_data/original_imgs_and_mask/original_imgs"
# og_csvp = "/fast_data_e2e11/piyush/fake_data/fake_data_csvs/og_image_model_scores_neg.csv"
test_csv_p = "/fast_data_e2e_1/cxr/qxr_ln_data/LN_test/combined_test_csv_updated_internal_test_13-08-24.csv"
mask_root = "/fast_data_e2e_1/cxr/qxr_ln_data/testing_data_masks/lung"

# model_output_df = pd.read_csv(og_csvp)
# model_output_df = model_output_df.set_index('filename')

# filenames_png = model_output_df.index.to_list()
test_df = pd.read_csv(test_csv_p)
png_paths = test_df.png_path.to_list()

def lnorm(arr):
    arr = arr - arr.mean()
    arr = arr / (arr.std() + 0.000001)
    return arr


def scale(arr):
    """scale an array between 0 and 1"""
    mn = arr.min()
    arr = arr - arr.min()
    if arr.max() == 0:
        return arr
    else:
        arr = arr/(arr.max() + arr.max()/10e10)
        return arr
part_size = int(len(png_paths)/4)

# start_idx = 0
# end_idx = part_size

# start_idx = part_size
# end_idx = 2*part_size -1 

start_idx = 2*part_size 
end_idx = 3*part_size -1 

# start_idx = 3*part_size 
# end_idx = len(png_paths)

with torch.no_grad():
    for png_path in pbar(png_paths[start_idx:end_idx]):
        filename = os.path.basename(png_path)
        output_path = f"{mask_root}/lung/{filename}"
        if os.path.exists(output_path): 
            continue
        img_path = png_path
        try: 
            og_img = cv2.imread(img_path, 0)/255
            img = lnorm(og_img)
            img = cv2.resize(img, (224, 224))
            # img_heart =cv2.resize(img, (960, 960))
            # img_heart = scale(img_heart)

        except Exception as e : 
            print(e)
            print("Can't load the image properly")
            continue
        ## Get the model outputs
        # heart_out = heart_model(torch.tensor(scale(img_heart)).float().unsqueeze(0).unsqueeze(0).cuda())
        thorax_out = torch.nn.functional.softmax(thorax_model(torch.tensor(lnorm(img)[None,None]).cuda().float())[0],dim=0)
        diaphragm_out = torch.nn.functional.softmax(diaphragm_model(torch.tensor(lnorm(img)[None,None]).cuda().float())[0],dim=0)

        ## Save the model outputs
        # heart_mask = (heart_out[1]['heart'].cpu().numpy()[0]>0.7)*255
        lmask = (thorax_out[1].cpu().numpy() > 0.7)*255
        rmask = (thorax_out[2].cpu().numpy() > 0.7)*255
        dlmask = (diaphragm_out[1].cpu().numpy() > 0.7)*255
        drmask = (diaphragm_out[2].cpu().numpy() > 0.7)*255

        # heart_mask = np.array(heart_mask, dtype=np.uint8)
        lmask = np.array(lmask, dtype=np.uint8)
        rmask = np.array(rmask, dtype=np.uint8)
        dlmask = np.array(dlmask, dtype=np.uint8)
        drmask = np.array(drmask, dtype=np.uint8)

        

        # heart_mask = cv2.resize(heart_mask, (og_img.shape[1], og_img.shape[0]))
        lmask = cv2.resize(lmask, (og_img.shape[1], og_img.shape[0]))
        rmask = cv2.resize(rmask, (og_img.shape[1], og_img.shape[0]))
        dlmask = cv2.resize(dlmask, (og_img.shape[1], og_img.shape[0]))
        drmask = cv2.resize(drmask, (og_img.shape[1], og_img.shape[0]))

        ##save the imgs

        cv2.imwrite(f"{mask_root}/lmask/{filename}", lmask)
        cv2.imwrite(f"{mask_root}/rmask/{filename}", rmask)
        cv2.imwrite(f"{mask_root}/dlmask/{filename}", dlmask)
        cv2.imwrite(f"{mask_root}/drmask/{filename}", drmask)
        # output_path = f"{mask_root}/heart/{filename}"
        # cv2.imwrite(output_path, heart_mask)