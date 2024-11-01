# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_utils.ipynb.

# %% auto 0
__all__ = ['hu_to_lung_window', 'image_grid', 'thumbnail', 'windowing', 'vis', 'load_deeplake_nodule_data', 'load_sitk_img',
           'import_module', 'locate_cls', 'clean_state_dict', 'mmcv_config_to_omegaconf']

# %% ../nbs/01_utils.ipynb 1
import pydoc
import PIL 
import SimpleITK as sitk
import numpy as np
from PIL import Image
from functools import partial
from collections import OrderedDict
from loguru import logger

# %% ../nbs/01_utils.ipynb 2
def image_grid(imgs, rows, cols):
    w,h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# %% ../nbs/01_utils.ipynb 3
def thumbnail(img, size=256):
    if not isinstance(img, PIL.Image.Image): img = Image.fromarray(img)
    w, h = img.size
    ar = h/w 
    return img.resize((size, int(size*ar)))

# %% ../nbs/01_utils.ipynb 4
def windowing(ww, wl):
    low = wl - ww/2
    high = wl + ww/2
    
    def _window(img):
        out = img.copy()
        out = (out-low)/(high-low)
        out = np.clip(out, 0, 1)
        return out 
    return _window

# %% ../nbs/01_utils.ipynb 5
hu_to_lung_window = windowing(1600, -600)

# %% ../nbs/01_utils.ipynb 6
def vis(img, size, window=True, seed=42):
    if window: 
        img = hu_to_lung_window(img)
        img = np.uint8(img*255)
    np.random.seed(seed)
    x = np.random.randint(img.shape[0], size=25)
    x.sort()
    img = [thumbnail(img[i], size) for i in x]
    return image_grid(img, 5, 5)

# %% ../nbs/01_utils.ipynb 7
def load_deeplake_nodule_data(ds, idx):
    f = {}
    keys = ["images", "boxes", "labels", "series_id", "mask", "spacing"]
    for key in keys:
        if key in ds.tensors.keys():
            f[key] = ds[key][idx].numpy()
        if key == "series_id": f[key] = f[key][0]        
    return f

# %% ../nbs/01_utils.ipynb 8
def load_sitk_img(series_path, series_id=None):
    img = series_path if isinstance(series_path, sitk.Image) else sitk.ReadImage(series_path)
    oimg = {}
    oimg["images"] = sitk.GetArrayFromImage(img)
    oimg["spacing"] = img.GetSpacing()[::-1]
    oimg["series_id"] = series_id if series_id is not None else ""
    return oimg 

# %% ../nbs/01_utils.ipynb 10
def import_module(d, parent=None, **default_kwargs):
    # copied from
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    try:
        if parent is not None:
            module = getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034
        else:
            module = pydoc.locate(object_type)(**kwargs)
    except Exception as e:
        logger.error(f"Cannot load {name}. Error: {str(e)}")
    return module

# %% ../nbs/01_utils.ipynb 11
def locate_cls(transforms: dict, return_partial=False):
    name = transforms["__class_fullname__"]
    targs = {k: v for k, v in transforms.items() if k != "__class_fullname__"}
    try:
        if return_partial:
            transforms = partial(pydoc.locate(name), **targs)
        else:
            transforms = pydoc.locate(name)(**targs)
    except Exception as e:
        logger.error(f"Cannot load {name}. Error: {str(e)}")
    return transforms

# %% ../nbs/01_utils.ipynb 12
def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing module. prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name.replace("model.", "")] = v
    return cleaned_state_dict

# %% ../nbs/01_utils.ipynb 13
def mmcv_config_to_omegaconf(cfg):
    from mmengine.config import ConfigDict
    from omegaconf import OmegaConf
    new_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, ConfigDict):
            v = v.to_dict()
        elif isinstance(v, list):
            v = [i.to_dict() if isinstance(i, ConfigDict) else i for i in v]
        new_cfg[k] = v
    cfg2 = OmegaConf.create(new_cfg)
    return cfg2
