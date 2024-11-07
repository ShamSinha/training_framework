# from omegaconf import DictConfig
# import importlib
# import copy
# from tqdm import tqdm

# # import model_files
# from qxr_utils.dicom.utils import get_array_from_dicom
# from qxr_utils.image.transforms import scale, resize
# from torch.utils.data import DataLoader
# import torch.utils.data as data
# import torch.nn.functional as fn


# import sys

# import cv2
# import matplotlib.pyplot as plt
# import pandas as pd
# import os

# import torch
# from torch.utils.data import Subset
# import numpy as np


# test_df = pd.read_csv("/fast_data_e2e_1/cxr/qxr_ln_data/LN_test/combined_test_csv_updated_internal_test_13-08-24.csv")


# ## helper functions 

# class ScanD(data.Dataset):
#     def __init__(self, imdict, imsize=960):
#         self.imdict = imdict
#         self.imsize = imsize
#         self.indices = list(self.imdict.keys())

#     def __getitem__(self, index):
#         imid = self.indices[index]
#         idx, im = self._get_im(imid)
#         return {"id": idx, "input": im}

#     def __len__(self):
#         return len(self.indices)

#     def _get_im(self, idx):
#         try:
#             impath = self.imdict[idx]
#             ext = os.path.splitext(impath)[-1]
#             if ext == ".dcm":
#                 try:
#                     im = get_array_from_dicom(impath)
#                 except Exception as e:
#                     #                     print(e)
#                     im = None
#             else:
#                 im = cv2.imread(impath, 0)
#             if im is not None:
#                 im = resize(self.imsize, self.imsize)(im)
#                 im = scale(im)
#                 im = torch.Tensor(im).reshape(1, self.imsize, self.imsize)
#                 return idx, im
#             else:
#                 return "none", torch.zeros(1, self.imsize, self.imsize)
#         except Exception as e:
#             print(e)
#             print(idx)
            
            
# def dict_to_df(preds):
#     df = pd.DataFrame.from_dict(preds)
#     df = df.transpose().reset_index()
#     df = df.rename(columns={"index": "filename"})
#     return df

# def run_model(model, dloader, names, seg=False, save_dir=None):
    
#     if seg:
#         assert save_dir, "save_dir not provided"
    
#     preds = {}
#     for batch in tqdm(dloader):
#         with torch.no_grad():
#             ids = batch['id']
#             inputs = batch['input']
        
#             out = model(inputs.to("cuda:0"))

#             cls_out = out[0]
#             seg_out = out[1]
            
#             for tag in names:
#                 tdata = None
#                 sdata = None
                
                    
#                 if seg and (tag in seg_out):
#                     # sdata = fn.softmax(seg_out[tag].cpu(), dim=1)
#                     os.makedirs(os.path.join(save_dir, tag), exist_ok=True)

#                 tdata = cls_out[tag]
#                 sdata = seg_out[tag]

#                 for i, idx in enumerate(ids):
#                     if not idx in preds:
#                         preds[idx] = {}
                        
#                     if not (tdata is None):
#                         preds[idx].update({tag: tdata[i].item()})
                        
#                     if not (sdata is None):
#                         s_out = sdata[i].detach().cpu().numpy()
#                         binary_mask = (s_out > 0.3).astype(np.uint8) * 255
#                         seg_max = s_out.max().item()
                        
#                         preds[idx].update({f"{tag}_seg": seg_max})
#                         cv2.imwrite(f"{save_dir}/{tag}/{idx}.png", binary_mask)   
                        
                        
    
#     return dict_to_df(preds)

# ## define start and end idx 
# start_idx = int(sys.argv[1])
# end_idx = int(sys.argv[2])
# process_number = sys.argv[3]  
  

# ## prepare dataset / dataloader 
# png_dict = {}
# for id , row in test_df[start_idx:end_idx].iterrows():
#     filename = row['filename']
#     png_path = row['png_path']
#     png_dict[filename] = png_path
    
# dset = ScanD(png_dict, 960)
# dloader = DataLoader(dset, batch_size=12, num_workers=4)


# ## load mass model 
# model = torch.jit.load('/fast_data_e2e11/abhishek/qureai/packages/python/qxr/traces_ts/v4_mass_upgrade_cuda.ts', map_location='cuda')


# ## run mass model save df 
# pred_df = run_model(model, dloader, ['mass'], seg=True, save_dir="/fast_data_e2e_1/cxr/qxr_ln_data/testing_data_masks")

# output_filename = f"/fast_data_e2e_1/cxr/qxr_ln_data/testing_data_masks/mass/mass_scores{process_number}.csv"
# pred_df.to_csv(output_filename, index=False)


