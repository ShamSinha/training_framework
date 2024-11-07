import numpy as np 
import pandas as pd
import copy
from loguru import logger
from pathlib import Path
from voxdet.infer import RetinaInfer
from voxdet.tfsm.med import AddLungCache
from voxdet.utils import load_sitk_img
from voxdet.tfsm.utils import corner_2_chwd
from safetensors.numpy import save_file

# tt =  [AddLungCache(cache_dir="/cache/datanas1/qct-nodules/nifti_with_annots/lung_mask_cache/", 
#                     model_ckpt="resources/unet_r231-d5d2fc3d_v0.0.1.pth"), 
#        CropLung(model_ckpt=None),
       
#       ]
# tt = Compose(tt)

def transforms():
    from voxdet.utils import locate_cls
    from torchvision.transforms import Compose

    resolution = (1.0, 1.0, 1.0)
    test_transforms = [
        dict(__class_fullname__ = "voxdet.tfsm.standard.StandardT",
             src_mode="yxzhwd", 
             img_src_mode="zyx"),
        dict(__class_fullname__="voxdet.tfsm.med.Resample", 
              req_spacing=resolution),
        dict(__class_fullname__="voxdet.tfsm.med.AddLungCache", 
              cache_dir="/cache/datanas1/qct-nodules/nifti_with_annots/lung_mask_cache/", 
              model_ckpt="/home/users/vanapalli.prakash/repos/qct_nodule_detection/resources/unet_r231-d5d2fc3d_v0.0.1.pth"),
        dict(__class_fullname__="voxdet.tfsm.med.CropLung",
             req_spacing=resolution)
        ]
    transforms = Compose([locate_cls(i) for i in test_transforms])
    return transforms

tt = transforms()

CHECKPOINT_DET = "resources/exp_150/exp1.3.5_v2_no_rw.pth"
model = RetinaInfer(CHECKPOINT_DET)
model.transforms.transforms.insert(1, AddLungCache(cache_dir="/cache/datanas1/qct-nodules/nifti_with_annots/lung_mask_cache/", 
                                                    model_ckpt="resources/unet_r231-d5d2fc3d_v0.0.1.pth"))

save_dir = Path("/home/users/vanapalli.prakash/safe_ds/pseudo/")
scans_root = "/cache/datanas1/qct-nodules/nifti_with_annots/nlst/"
assert Path(scans_root).exists()
scans = list(Path(scans_root).glob("*.nii.gz"))
in_train = pd.read_csv("data1.4/v000/folds.csv")["scan_name"].unique()
scans = [i for i in scans if i.stem[:-4] not in in_train]
print(f"total_scans: {len(scans)}")

# Only keep scans which doesn't have annotations 
# dsb = read_nodules_csvs(["data1.3.4/v001/test/dsb_test.csv", "data1.3.4/v001/train/dsb_val.csv"])
# dsb_scan_names = dsb["scan_name"].unique().tolist()
# scans = [i for i in scans if i.stem[:-4] not in dsb_scan_names]

count=0
for num, scan in enumerate(scans):
    save_loc = save_dir/(scan.stem[:-4]+".safetensors")
    #if save_loc.exists(): continue 
    img = load_sitk_img(scan.as_posix(), scan.stem[:-4])
    timg = copy.deepcopy(img)
    nimg = model(img)
    boxes = nimg["boxes"][nimg["scores"]>0.95]
    boxes[:, 3:] = np.ceil(boxes[:, 3:])
    boxes[:, :3] = np.floor(boxes[:, :3])
    
    
    if boxes.shape[0] > 0:
        cboxes = corner_2_chwd(boxes)[:, [1, 2, 0, 4, 5, 3]] #zyxdhw => yxzhwd #012345 #120453
        ib = (cboxes[:, 3:][:, [2, 1, 0]] >= timg["images"].shape).sum(1)
        timg["boxes"] = cboxes[~np.bool_(ib)]
        
        timg["labels"] = np.zeros(timg["boxes"].shape[0]).astype(np.int8)
        #del img["spacing"]
        #img["images"] = nimg["images"]
        timg["images"] = timg["images"].astype(np.int16)
        timg["spacing"] = np.asarray(timg["spacing"])
        timg = tt(timg)
        timg["boxes"] = timg["boxes"][~np.bool_((timg["boxes"]<0).sum(1))]
        if (timg["boxes"]<0).sum(): breakpoint()
        #['boxes', 'images', 'labels', 'lung_box', 'spacing', 'volume']
        del timg["series_id"]
        del timg["lung_mask"]
        #img["boxes"] = img["boxes"][[1, 2, 0, 4, 5, 3]] #zyxdhw -> yxzhwd
        save_file(timg, save_loc)
        logger.info(f"{count}-{num}-{len(scans)} stored")
        count+=1
        
     

