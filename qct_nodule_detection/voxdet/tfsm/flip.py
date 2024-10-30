# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/tfsm/08_random_flip.ipynb.

# %% auto 0
__all__ = ['flip3d', 'RandFlip']

# %% ../../nbs/tfsm/08_random_flip.ipynb 2
import numpy as np
import fastcore.all as fc
from typing import List
from .standard import BaseT

# %% ../../nbs/tfsm/08_random_flip.ipynb 21
def flip3d(img:np.ndarray, boxes:np.ndarray, flip_axis:int):
    """img zyx/bzyx format, bboxes in zyxzyx format and flip_axis can be one of [0, 1, 2]
    flip_axis: 0: depth flip, 1: horizontal flip, 2: vertical flip
    """
    assert flip_axis in [0, 1, 2], f"flip axis cannot be {flip_axis}. need to be one of [0, 1, 2]"
    assert img.ndim in [3, 4], f"works only for 3D images. dimensions can be 3 or 4. Incase 4, 1st dim is channels"
    fa = flip_axis+1 if img.ndim > 3 else flip_axis
    flipped_img = np.flip(img, axis=fa)
    ims = img.shape[1:] if img.ndim > 3 else img.shape
    idx_pairs = [[0, 3], [1, 4], [2, 5]]
    boxes[:, idx_pairs[flip_axis]] = ims[flip_axis] - boxes[:, idx_pairs[flip_axis][::-1]]
    return flipped_img, boxes

# %% ../../nbs/tfsm/08_random_flip.ipynb 46
class RandFlip(BaseT):
    def __init__(self, axis:List[int]=None, p:List[float]=None):
        """
        flip the image about given axes with given probabilities
        order of axes will be the order in which the flip transforms
        will be applied
        """
        super().__init__()
        fc.store_attr()
        if len(axis) != len(p): raise ValueError(f"len of axis and p should be same. Got axis={axis} and p={p} ")
    __repr__ = fc.basic_repr(flds="axis, p")
    
    def apply(self, img: dict):
        assert "images" in img.keys()
        fimg = img["images"].copy()
        boxes = img["boxes"].copy()
        
        if isinstance(fimg , np.ndarray) :
            fimg = [fimg]
            boxes = [boxes]
            
        for i in range(len(fimg)) :
            for fa, fp in zip(self.axis, self.p):
                if np.random.random() <= fp:
                    fimg[i] , boxes[i] = flip3d(img=fimg[i], boxes=boxes[i], flip_axis=fa)
        
        if len(fimg) == 1 :
            nimg = {"images":fimg[0], "boxes":boxes[0]}
        else :
            nimg = {"images":fimg, "boxes":boxes}
            
        for i in img.keys(): 
            if i not in ["images", "boxes"]: nimg[i] = img[i]
        return nimg
