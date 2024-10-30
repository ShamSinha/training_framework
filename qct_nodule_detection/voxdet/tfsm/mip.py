# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/tfsm/03_mip.ipynb.

# %% auto 0
__all__ = ['get_mip_array', 'MIP']

# %% ../../nbs/tfsm/03_mip.ipynb 1
import math
import numpy as np 
import fastcore.all as fc

from .standard import BaseT

# %% ../../nbs/tfsm/03_mip.ipynb 8
def get_mip_array(img, stride:int=1, num_slices:int=5, mode:str="max"):
    assert img.ndim == 3 or 4, f"Expected NumPy array to have 3 or 4 dims. Got shape {img.shape}"
    
    func = {"max": np.amax, "min": np.amin, "mean": np.mean}
    if mode not in func:
        raise NotImplementedError

    slices_per_mip = []
    
    n_mip_slices = math.ceil(img.shape[0]/stride)
    np_mip = np.empty((n_mip_slices, img.shape[1], img.shape[2]), img.dtype)

    # can we vectorize this further?
    for i in range(n_mip_slices):
        start = i*stride
        
        end = min(start+num_slices, img.shape[0])
        np_mip[i] = func[mode](img[start:end], 0)
        slices_per_mip.append(end - start)

    return np_mip.astype(np.float64)

# %% ../../nbs/tfsm/03_mip.ipynb 12
class MIP(BaseT):
    def __init__(self, num_slices: int = 5, stride: int = 1, mode:str = "max", return_stacked_img:bool = True, mip_channel:int = 0 ):
        """create MIP numpy array from given image numpy array"""
        fc.store_attr()
        super().__init__()
    __repr__ = fc.basic_repr(flds="num_slices, mode, stride, return_stacked_img, mip_channel")
    
    def apply(self, img: dict):
        assert all([i in list(img.keys()) for i in ["images"]])
        fimg, nimg = img["images"], {}
        nimg["images"] = self.apply_image(fimg)
        for i in img.keys(): 
            if i not in nimg.keys(): nimg[i] = img[i]
        return nimg
    
    def apply_image(self, img: np.asarray):
        if img.ndim == 3 :
            mip = get_mip_array(img, stride=self.stride, num_slices=self.num_slices, mode=self.mode)
            stacked_img = np.stack((img, mip), axis=0)
        if img.ndim == 4 :
            mips = []
            for ch in self.mip_channel : 
                mip = get_mip_array(img[ch], stride=self.stride, num_slices=self.num_slices, mode=self.mode)
                mips.append(mip)
            mips = np.stack(mips, axis = 0)
            stacked_img = np.concatenate((img, mips), axis=0)
        
        if self.return_stacked_img: return stacked_img
        return mip
        
    def reverse_apply(self, img: dict): return img