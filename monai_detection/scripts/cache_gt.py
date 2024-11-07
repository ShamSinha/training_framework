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
from qct_utils.apps.nod_dia import get_major_minor_coordinates_3d_nodule
from qct_utils.utils import load_txt

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
        # csv_loc =   "/home/users/shubham.kumar/projects/recist/recist_annotation_first_readers.csv"
        # df = pd.read_csv(csv_loc)
        # series_ids = df.scan_name.unique()

        # csv_loc = f"/home/users/shubham.kumar/projects/qct_data/qct_data_status/det/data/{dataset}.csv"
        # df = pd.read_csv(csv_loc)

        csv_loc  = f"/home/users/shubham.kumar/projects/qct_data/qct_data_status/{dataset}/data/merged.csv"
        df = pd.read_csv(csv_loc)

        series_ids = load_txt("/home/users/shubham.kumar/projects/qct_data/qct_data_status/recist_segmed/data/test.txt")


        target_lesions = ["TL1", "TL2", "TL3", "TL4", "TL5"]
        non_target_lesions = ["NTL"]
        new_lung_lesions = ["NLL"]

        def merge_into_one_lesion_type(x):
            if "recist_lesion_type" in eval(x).keys():
                present_lesions = eval(x)["recist_lesion_type"].split("|")
                if len(present_lesions) == 1 or len(set(present_lesions)) == 1:
                    return present_lesions[0]
                if len(present_lesions) == 2:
                    flagged_values = list(set(present_lesions).intersection(target_lesions))
                    if len(flagged_values) > 0:
                        return flagged_values[0]
                    else:
                        return "NTL"
            else:
                return None
            
        df["meta"] = df["meta_union"].apply(
            lambda x: str({"recist_lesion_type": merge_into_one_lesion_type(x)})
        )

        df = df.rename(columns={"annot_union": "annot"})
        df = df.sort_values(by="scan_name")
        df["counter"] = df.groupby("scan_name").cumcount() + 1
        df["annot_id"] = df["scan_name"] + "_" + df["counter"].astype(str)
        df.drop(columns=["counter"], inplace=True)
        df["annotated_by"] = df["rads"].apply(lambda x : f"{x}_readers")  

        # df.to_csv(csv_loc , index =False)
        # series_ids = df[df["status"] == phase]["scan_name"].unique()

        # meta_root = get_qct_collection(collections[dataset])
        df_spacing = pd.read_csv("/home/users/shubham.kumar/projects/qct_data/qct_data_status/recist_segmed_metastatic/data/spacing.csv")
        series_ids = df_spacing["scan_name"].unique()
        meta_root = get_qct_collection("recist_segmed")
        ds = CTAnnotLoader(scans_root=meta_root, csv_loc=df, series_ids=series_ids,dummy_scans=True)

        for sid in tqdm(series_ids ,desc= dataset):
            # img_path = (
            #     f"/cache/fast_data_nas8/qct/shubham/det_ctscan/{dataset}/{sid}.pt"
            # )  
            save_path = f"/cache/fast_data_nas8/qct/shubham/det_gt_annot/{dataset}/{sid}.pt"
            
            # if os.path.exists(img_path):
            if True :
                file = {}
                file["bbox"] = []
                file["meta"] = []
                file["rads"] = []
                try :
                    ctscan = ds[sid]
                    # img = torch.load(img_path)
                    # spacing = ITKDIM3D.from_np(img["spacing"])
                    spacing = ITKDIM3D(**eval(df_spacing[df_spacing["scan_name"] == sid].spacing.values[0]))

                    for annot in ctscan.nodules:
                        annot.gt.annot.bbox = annot.gt.annot.mask.to_boxc()
                        out = get_major_minor_coordinates_3d_nodule(annot.gt.annot.mask.array , spacing )
                        
                        lad = out.long_axis_len
                        sad = out.short_axis_len
                        meta = annot.gt.meta.model_dump()
                        rads = annot.gt.annotated_by
                        
                        meta['short_axis_diameter'] = sad
                        meta['long_axis_diameter'] = lad
                        bbox = annot.gt.annot.bbox.xcyczcwhd[[1, 0, 2, 4, 3, 5]]
                        file["bbox"].append(bbox.tolist())
                        file["meta"].append(meta)
                        file["rads"].append(rads)

                except KeyboardInterrupt :
                    break
                except Exception as e :
                    logger.debug(e)
                    continue
                
                if len(file["bbox"]) > 0 : 
                    torch.save(file, save_path )
                
