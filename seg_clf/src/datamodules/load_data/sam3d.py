from safetensors import safe_open
import glob
from typing import List
import pandas as pd
import os

def _load_data(directory: str, phase) :
    li = []

    if phase == "train" :
        data_type = "Tr"
    if phase == "val" :
        data_type = "Val"

    # if ${path}/labelsTr exists, search all .nii.gz
    d = os.path.join(directory, f'labels{data_type}')
    if os.path.exists(d):
        for name in os.listdir(d):
            temp_data = {}
            base = os.path.basename(name).split('.nii.gz')[0]
            label_path = os.path.join(directory, f'labels{data_type}', f'{base}.nii.gz')
            temp_data["label"] = label_path
            temp_data["image"] = label_path.replace('labels', 'images')
      
            li.append(temp_data)
    return li
    

def videomae_data_loader(
    directory: str,
    phase: str,
    frac: float,
):
    """
    Returns:
    --------
    List of dicts. Each dicts have information about that datapoint.
    """

    return _load_data(directory, phase)