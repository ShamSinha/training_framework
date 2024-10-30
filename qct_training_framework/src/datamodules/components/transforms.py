import random
from ast import literal_eval

import numpy as np
import torch
from monai.transforms import RandomizableTransform


class RandomSafeCrop(RandomizableTransform):
    def __init__(self, prob: float = 1.0) -> None:
        RandomizableTransform.__init__(self, prob)

    def __call__(self, data_dict: dict):
        self.randomize(None)
        if not self._do_transform:
            return data_dict
        scan_im = data_dict["image"]
        mask_im = data_dict["label"]

        zz, yy, xx = np.where(mask_im)
        x_min, x_max = np.min(xx), np.max(xx)
        y_min, y_max = np.min(yy), np.max(yy)
        z_min, z_max = np.min(zz), np.max(zz)
        s_z, s_y, s_x = mask_im.shape

        x1 = random.randint(0, x_min)
        x2 = random.randint(x_max, s_x)
        y1 = random.randint(0, y_min)
        y2 = random.randint(y_max, s_y)
        z1 = random.randint(0, z_min)
        z2 = random.randint(z_max, s_z)

        crop_scan_im = scan_im[z1:z2, y1:y2, x1:x2]
        crop_mask_im = mask_im[z1:z2, y1:y2, x1:x2]

        # Rarely some of the these dims gets zero.
        if np.prod(crop_mask_im.shape) != 0:
            data_dict["image"] = crop_scan_im
            data_dict["label"] = crop_mask_im

        return data_dict


def change_affine_matrix(data_dict):
    affine_mat = data_dict["image"].affine
    affine_mat[0][0] = literal_eval(data_dict["spacing"])["z"]
    affine_mat[1][1] = literal_eval(data_dict["spacing"])["y"]
    affine_mat[2][2] = literal_eval(data_dict["spacing"])["x"]
    affine_mat = data_dict["label"].affine
    affine_mat[0][0] = literal_eval(data_dict["spacing"])["z"]
    affine_mat[1][1] = literal_eval(data_dict["spacing"])["y"]
    affine_mat[2][2] = literal_eval(data_dict["spacing"])["x"]
    return data_dict


def change_mask_to_binary(data_dict):

    data_dict["label"] = data_dict["label"].set_array(data_dict["label"].to(torch.int8))
    return data_dict
