import torch
from qct_utils.ctscan.ct_loader import CTAnnotLoader
import fastcore.all as fc
import pandas as pd
import numpy as np
import os
from loguru import logger
from voxdet.create_ds import ctscan2dict
from tqdm.auto import tqdm
from qct_data.series_meta import get_qct_collection
import argparse
from qct_utils.schema.dim import ITKDIM3D
from qct_utils.apps.nodule_diameter import get_major_minor_coordinates_3d_nodule


parser = argparse.ArgumentParser(description="Save CTscan to mitigate faster inference")

if __name__ == "__main__":

    # parser.add_argument("--phase", type=str, help="dataset splits:  train/val/test")
    parser.add_argument(
        "datasets", type=str, nargs="+", help="A list of datasets to be processed."
    )
    args = parser.parse_args()
    # phase = args.phase
    datasets = args.datasets

    collections = {"wcg" : "wcg_fda_qct" , "internal" : "qure_internal" , "qxr_fda" : "wcg_fda_qxr"}
    for dataset in datasets :
        if dataset not in collections.keys() : 
            collections.update({dataset : dataset})

    for dataset in datasets:
        fc.Path.mkdir(fc.Path(f"/cache/fast_data_nas8/qct/shubham/det_gt_annot/{dataset}/"), exist_ok=True)
        csv_loc = f"/home/users/shubham.kumar/projects/interns/qct_data/qct_data_status/det/data/{dataset}.csv"
        df = pd.read_csv(csv_loc)
        df["annotated_by"] = df["rads"].apply(lambda x : f"{x}_readers")
        df.to_csv(csv_loc , index =False)
        # series_ids = df[df["status"] == phase]["scan_name"].unique()
        series_ids = df["scan_name"].unique()

        
        meta_root = get_qct_collection(collections[dataset])

        ds = CTAnnotLoader(scans_root=meta_root, csv_loc=csv_loc, series_ids=series_ids,dummy_scans=True)

        for sid in tqdm(series_ids ,desc= dataset):
            img_path = (
                f"/cache/fast_data_nas8/qct/shubham/det_ctscan/{dataset}/{sid}.pt"
            )                
            save_path = f"/cache/fast_data_nas8/qct/shubham/det_gt_annot/{dataset}/{sid}.pt"
            
            # if os.path.exists(save_path) :
            #     continue
            if os.path.exists(img_path):
                try :
                    file = torch.load(save_path)
                    file["rads"] = []
                    ctscan = ds[sid]
                    for annot in ctscan.nodules:
                        # annot.gt.annot.bbox = annot.gt.annot.mask.to_boxc()
                        # out = get_major_minor_coordinates_3d_nodule(annot.gt.annot.mask.array , spacing )
                        
                        # lad = out.long_axis_len
                        # sad = out.short_axis_len
                        # meta = annot.gt.meta.model_dump()
                        rads = annot.gt.annotated_by
                    
                        
                        # meta['short_axis_diameter'] = sad
                        # meta['long_axis_diameter'] = lad
                        # bbox = annot.gt.annot.bbox.xcyczcwhd[[1, 0, 2, 4, 3, 5]]
                        # file["bbox"].append(bbox.tolist())
                        # file["meta"].append(meta)
                        file["rads"].append(rads)
                except KeyboardInterrupt :
                    break
                except Exception as e :
                    logger.debug(e)
                    continue
                
                if len(file["bbox"]) > 0 : 
                    torch.save(file, save_path )
                
