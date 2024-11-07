import os
import torch
import argparse
import fastcore.all as fc
import pandas as pd
import numpy as np
from loguru import logger
from tqdm.auto import tqdm
from qct_data.series_meta import get_qct_collection
from qct_utils.ctscan.ct_loader import CTAnnotLoader
from qct_utils.utils import load_txt

parser = argparse.ArgumentParser(description="Save CTscan to mitigate faster inference")

if __name__ == "__main__":

    # parser.add_argument("--phase", type=str, help="dataset splits:  train/val/test")
    # parser.add_argument(
    #     "--qct_data_path", type=str, help="provide you latest qct_data repo path"
    # )
    # parser.add_argument(
    #     "--save_dir", type=str, help="provide dir path to save ctscan"
    # )
    parser.add_argument(
        "datasets", type=str, nargs="+", help="A list of datasets to be processed."
    )
    args = parser.parse_args()
    # phase = args.phase
    qct_repo_path = "/home/users/shubham.kumar/projects/qct_data"
    save_dir = "/cache/fast_data_nas8/qct/shubham/det_ctscan/"
    datasets = args.datasets

    collections = {
        "wcg": "wcg_fda_qct",
        "internal": "qure_internal",
        "qxr_fda": "wcg_fda_qxr",
    }
    for dataset in datasets:
        if dataset not in collections.keys():
            collections.update({dataset: dataset})

    for dataset in tqdm(datasets, desc=dataset):
        fc.Path.mkdir(fc.Path(os.path.join(save_dir , dataset)), exist_ok=True)

        # csv_loc = os.path.join(
        #     qct_repo_path, f"qct_data_status/{dataset}/data/merged.csv"
        # )
        # df = pd.read_csv(csv_loc)
        # series_ids = df["scan_name"].unique()

        # series_ids = load_txt("/home/users/shubham.kumar/projects/qct_data/qct_data_status/recist_segmed/data/test.txt")
        series_ids = pd.read_csv("/home/users/shubham.kumar/projects/qct_data/qct_data_status/recist_segmed_metastatic/data/spacing.csv")["scan_name"].unique()
        # meta_root = get_qct_collection(collections[dataset])
        meta_root = get_qct_collection("recist_segmed")
        ds = CTAnnotLoader(scans_root=meta_root, csv_loc=None, series_ids=series_ids)

        for sid in tqdm(series_ids):
            save_path = os.path.join(save_dir , dataset , f"{sid}.pt")
            if not os.path.exists(save_path):
                try:
                    ctscan = ds[sid]
                except Exception as e:
                    logger.debug(e)
                    continue

                img = {}
                spacing = ctscan.scan.spacing
                img["series_id"] = ctscan.series_instance_uid
                img["images"] = ctscan.scan.array.astype(np.int16)
                img["spacing"] = np.asarray([spacing.z, spacing.y, spacing.x])
                torch.save(img, save_path)
