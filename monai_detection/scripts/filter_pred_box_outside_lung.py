import numpy as np
import torch
from tqdm.auto import tqdm
import pandas as pd
import os
from voxdet.retina_test import convert2int
import argparse
from loguru import logger


def filter_boxes_by_mask(boxes, binary_mask):
    filtered_boxes = []
    keep_index = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        z1, y1, x1, z2, y2, x2 = box
        # Ensure coordinates are within the mask boundaries
        
        # Extract the sub-volume from the mask
        sub_volume = binary_mask[z1:z2, y1:y2, x1:x2]
        
        # Check if there's any overlap with the region of interest
        if np.sum(sub_volume) > 0:
            filtered_boxes.append(box)
            keep_index.append(i)
       
    return np.array(filtered_boxes) , keep_index


parser = argparse.ArgumentParser(description="Add extra key in ct cache to keep which pred bboxes")

if __name__ == "__main__":

    parser.add_argument("--phase", type=str, help="dataset splits:  train/val/test")
    parser.add_argument(
        "datasets", type=str, nargs="+", help="A list of datasets to be processed."
    )
    args = parser.parse_args()
    phase = args.phase
    datasets = args.datasets

    for dataset in tqdm(datasets) : 
        df = pd.read_csv(f"/home/users/shubham.kumar/projects/qct_data/qct_data_status/det/data/{dataset}_merged.csv")
        series_ids = df[df["status"] == phase]["scan_name"].unique()

        for sid in tqdm(series_ids) : 

            if os.path.exists(f"/cache/fast_data_nas8/qct/shubham/det_ctscan/{dataset}/{sid}.pt"):
                img = torch.load(f"/cache/fast_data_nas8/qct/shubham/det_ctscan/{dataset}/{sid}.pt",map_location = "cpu")
                                
                req_df = df[df["scan_name"] == sid]
                lung_mask_arr = np.load(f"/cache/fast_data_nas8/qct/shubham/lung_mask_cache_det/{dataset}/{sid}_lung_mask.npy")
                
                pred_box = convert2int(img["pred_boxes"])
                
                final_corrected_box , keep_index = filter_boxes_by_mask(pred_box, lung_mask_arr)
                
                if final_corrected_box.shape[0] != pred_box.shape[0] :
                    logger.debug(keep_index)
                    img["keep_index_rm_box_outside_lungmask"] = np.array(keep_index)
                
                    torch.save(img ,f"/cache/fast_data_nas8/qct/shubham/det_ctscan/{dataset}/{sid}.pt")
                                    