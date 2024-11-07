from torchvision.transforms import Compose
from qct_utils.cv_ops.windowing import Window
from voxdet.tfsm.voxt import ApplyWindowsChannelWise, PadIfNeeded
from voxdet.tfsm.mip import MIP
from voxdet.tfsm.standard import StandardT
from voxdet.tfsm.med import CropLung
from voxdet.infer import RetinaInfer
from copy import deepcopy
from qct_utils.schema.ct import CTScan
import numpy as np
from loguru import logger
from qct_data.series_meta import get_qct_collection
import torch
from  tqdm.auto import tqdm
import pandas as pd
import os

def get_model(device="cpu"):
    model = RetinaInfer(
        checkpoint_path="/cache/fast_data_nas8/qct/shubham/qct_det_ckpt/epoch=89_step=1530_val_mAR=0.641.ckpt",
        device=device,
    )
    image_transforms = Compose(
        [
            StandardT(src_mode="yxzhwd", img_src_mode="zyx"),
            CropLung(margin=(2, 5, 5), device=device),
            ApplyWindowsChannelWise(
                renorm=True,
                windows=[Window(ww=2000, wl=-700), Window(ww=1000, wl=-400)],
            ),
            PadIfNeeded(sd=32),
            MIP(
                num_slices=5,
                stride=1,
                mode="max",
                return_stacked_img=True,
                mip_channel=[0],
            ),
        ]
    )
    model.transforms = image_transforms
    return model


if __name__ == "__main__":

    model = get_model(device="cuda:0")
    # scans_meta = get_qct_collection("dedomena_non_cancer")

    # save_dir = "/cache/fast_data_nas8/qct/shubham/dedomena_non_cancer"

    # file_path = "/home/users/vanapalli.prakash/data_studies/qct_data/qct_data_status/dedomena_non_cancer/vs.txt"
    # lines = []
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()

    # series_ids = [line.strip() for line in lines]  

    save_dir = "/cache/fast_data_nas8/qct/shubham/recist_segmed_annot"
    scans_meta = get_qct_collection("recist_segmed")
    pids = pd.read_csv("/home/users/shubham.kumar/projects/recist/batch1_6_recist_segmed_patient.csv").PatientID.values
    dd  =pd.read_csv("/home/users/shubham.kumar/projects/recist_wip/data/recist_segmed.csv")
    
    for pid in tqdm(pids) : 
        series_ids = eval(dd[dd["patient_id"] == pid ].series_id.values[0])
        for sid in tqdm(series_ids,leave=True) :
            if os.path.exists(os.path.join(save_dir , f"{sid}.pt")) :
                continue
            series_dict = scans_meta.find_one({"_id": sid})

            try : 
                ctimg = CTScan.load(
                    series_dict, readtype="dcm", scan_type="chestct", gaussian_smoothing_sigma=None
                )
                spacing = ctimg.spacing

                img = {}
                img["series_id"] = sid
                img["images"] = ctimg.array
                img["spacing"] = np.asarray([spacing.z, spacing.y, spacing.x])

                img_out = model(deepcopy(img))
                pred_boxes = img_out["boxes"]
                pred_scores = img_out["scores"]

                keep = pred_scores >= 0.9
                final_boxes, final_scores = pred_boxes[keep], pred_scores[keep]


                file = {}
                file["pred_boxes"] = final_boxes
                file["pred_scores"] = final_scores

                torch.save(file , os.path.join(save_dir , f"{sid}.pt"))
            except KeyboardInterrupt: 
                break
            except Exception as e :
                print(e)



