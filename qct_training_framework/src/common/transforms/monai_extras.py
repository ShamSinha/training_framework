import random
from typing import Any, Dict, List, Sequence, Union, Tuple, Optional

import numpy as np
import SimpleITK as sitk
import torch
import time
import os
from loguru import logger
from monai.transforms import DivisiblePadd, RandomizableTransform, Transform, Resize
from scipy import ndimage as ndi

from qct_utils.cv_ops.mip import create_MIP_numpy
from qer.utils.imageoperations.resampler import load_raw_sitk_img


# from qct_utils.schema.dim import ITKDIM3D, ITKDIM2D, BBox3D, Mask
# from qct_utils.apps.img_series_classifier.utils import BodyCrop
from torch.nn.functional import interpolate
from skimage.transform import rescale
from pydicom import dcmread
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F

from ...datamodules.load_data.utils import get_bin_indices
import h5py
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import (
    ball,
    disk,
    binary_erosion,
    binary_dilation,
    binary_closing,
    binary_opening,
    erosion,
    dilation,
    closing,
)

from scipy.ndimage import binary_fill_holes
from qer.ai.cta_lvo.preprocessing_utils import get_largest_island
import cv2
from safetensors import safe_open
from copy import deepcopy

WINDOWS = {
    # [window_level, window_width]
    "Bone": [500, 2000],
    "Lung": [-600, 1200],
    "LungA": [-600, 1600],
    "LungB": [-600, 900],
    "LungC": [-800, 400],
    "Abdomen": [40, 400],
    "Brain": [30, 70],
    "Soft Tissue": [50, 350],
    "Liver": [60, 160],
    "Mediastinum": [50, 500],
    "Stroke": [30, 30],
    "CTA": [170, 600],
    "Custom LungA": [-200, 160],
    "Custom LungB": [0, 900],
    "Custom LungC": [-900, 600],
    "Custom LungD": [-745, 1145],
    "bleed_1": [18, 12],
    "bleed_2": [14, 27],
    "bleed_3": [40, 40],
    "brain_window": [40, 80],
    "blood_window": [50, 175],
    "bone_window": [500, 3000],
}


# class Resize3D(Transform):

#     """
#     Resize3D is a PyTorch-based data augmentation transform for 3D medical images that
#     resizes both the image and the associated segmentation label maps to a specified
#     spatial size. This transform is useful for ensuring that all images in a dataset have
#     the same size, which can simplify the training process.

#     Args:
#         spatial_size (Union[Sequence[int], int]): The target spatial size of the image and
#             segmentation label maps. If a single integer is provided, it is interpreted as
#             the size of a cubic volume.
#         image_key (str): The key for the image tensor in the input dictionary. Defaults to
#             "image".
#         label_keys (Union[List[str], str]): The keys for the label maps in the input dictionary.
#             If a single string is provided, it is interpreted as the key for a single label map.
#             Defaults to "label".
#     Returns:
#         Dict[str, torch.Tensor]: A dictionary containing the resized image and segmentation label maps.
#     """

#     def __init__(
#         self,
#         spatial_size: Union[Sequence[int], int],
#         image_key: str = "image",
#         label_keys: Union[List[str], str] = "label",
#     ):
#         """
#         Initializes a new instance of the Resize3D transform.
#         """
#         self.image_key = image_key
#         self.label_keys = label_keys

#         self.size = tuple(spatial_size)
#         self.interpolation = "trilinear"

#     def __call__(self, data: Dict):
#         """
#         Applies the Resize3D transform to the input data.

#         Args:
#             data (Dict[str, torch.Tensor]): A dictionary containing the input image and segmentation
#                 label maps.

#         Returns:
#             Dict[str, torch.Tensor]: A dictionary containing the resized image and segmentation label maps.
#         """
#         data[self.image_key] = (
#             data[self.image_key].unsqueeze(0).unsqueeze(0)
#             if data[self.image_key].ndim == 3
#             else data[self.image_key]
#         )
#         data[self.image_key] = (
#             interpolate(
#                 data[self.image_key],
#                 size=self.size,
#                 mode=self.interpolation,
#             )
#             .squeeze(0)
#             .squeeze(0)
#         )
#         unique_labels = {}
#         binary_masks = {}
#         for key in self.label_keys:
#             unique_labels[key] = torch.unique(data[key])[1:]
#             data[key] = data[key].to(torch.int8)
#             binary_masks[key] = torch.stack(
#                 [((data[key] == label) * 1.0) for label in unique_labels[key]], dim=0
#             ).unsqueeze(
#                 0
#             )  # create binary masks for each label
#             resized_mask = interpolate(
#                 binary_masks[key], size=self.size, mode=self.interpolation
#             ).squeeze(0)
#             for i in range(len(unique_labels[key])):
#                 resized_mask[i] = (resized_mask[i] >= 0.5) * unique_labels[key][i]
#             data[key], _ = torch.max(resized_mask, dim=0, keepdim=False)
#             data[key] = data[key].to(torch.int8)
#         return data


class Resize3D(Transform):

    """
    Resize3D is a PyTorch-based data augmentation transform for 3D medical images that
    resizes both the image and the associated segmentation label maps to a specified
    spatial size. This transform is useful for ensuring that all images in a dataset have
    the same size, which can simplify the training process.

    Args:
        spatial_size (Union[Sequence[int], int]): The target spatial size of the image and
            segmentation label maps. If a single integer is provided, it is interpreted as
            the size of a cubic volume.
        image_key (str): The key for the image tensor in the input dictionary. Defaults to
            "image".
        label_keys (Union[List[str], str]): The keys for the label maps in the input dictionary.
            If a single string is provided, it is interpreted as the key for a single label map.
            Defaults to "label".
    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the resized image and segmentation label maps.
    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        image_key: str = "image",
        label_keys: Optional[Union[List[str], str]] = None,
    ):
        """
        Initializes a new instance of the Resize3D transform.
        """
        self.image_key = image_key
        self.label_keys = label_keys

        self.size = tuple(spatial_size)
        self.interpolation = "trilinear"

        self.resize_labels = Resize(
            spatial_size=spatial_size,
            size_mode="all",
            mode="nearest",
            align_corners=None,
            anti_aliasing=False,
        )

    def __call__(self, data: Dict  ):
        """
        Applies the Resize3D transform to the input data.

        Args:
            data (Dict[str, torch.Tensor]): A dictionary containing the input image and segmentation
                label maps.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the resized image and segmentation label maps.
        """
        if data["data_type"] == "mae" :
            return data
        if self.size != data[self.image_key].shape[-3:] :
            
            data[self.image_key] = (
                data[self.image_key].unsqueeze(0).unsqueeze(0)
                if data[self.image_key].ndim == 3
                else data[self.image_key]
            )
            data[self.image_key] = (
                interpolate(
                    data[self.image_key],
                    size=self.size,
                    mode=self.interpolation,
                )
                .squeeze(0)
                .squeeze(0)
            )
        
            if self.label_keys is not None :
                for key in self.label_keys:
                    data[key] = data[key].unsqueeze(0).to(torch.int8)
                    data[key] = self.resize_labels(data[key]).squeeze(0)

        return data


class RandomSafeCropd(RandomizableTransform):
    """Often we add additional fixed margin to the image.

    This crops the (image, label) randomly with entire mask present
    """

    def __init__(
        self,
        prob: float = 1.0,
        image_key: str = "image",
        label_keys: Union[List[str], str] = "label",
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.image_key = image_key
        self.label_keys = label_keys

    def __call__(self, data_dict: dict):
        self.randomize(None)
        if not self._do_transform:
            return data_dict

        if isinstance(self.label_keys, str):
            mask_im = data_dict[self.label_keys]
            self.label_keys = [self.label_keys]
        else:
            mask_im = []
            for key in self.label_keys:
                mask_im.append(data_dict[key])

            mask_im = torch.stack(mask_im)
            mask_im, _ = torch.max(mask_im, dim=0, keepdim=False)

        mask_im_np = mask_im.cpu().numpy()  # convert PyTorch tensor to numpy array
        if len(np.unique(mask_im_np)) == 1:
            return data_dict
        zz, yy, xx = np.where(mask_im_np)

        x_min, x_max = np.min(xx), np.max(xx)
        y_min, y_max = np.min(yy), np.max(yy)
        z_min, z_max = np.min(zz), np.max(zz)
        s_z, s_y, s_x = mask_im_np.shape

        x1 = random.randint(0, x_min)  # nosec
        x2 = random.randint(x_max, s_x)  # nosec
        y1 = random.randint(0, y_min)  # nosec
        y2 = random.randint(y_max, s_y)  # nosec
        z1 = random.randint(0, z_min)  # nosec
        z2 = random.randint(z_max, s_z)  # nosec

        scan_im = data_dict[self.image_key]
        crop_scan_im = scan_im[z1:z2, y1:y2, x1:x2]

        data_dict[self.image_key] = crop_scan_im
        for label_key in self.label_keys:
            label_im = data_dict[label_key]
            crop_label_im = label_im[z1:z2, y1:y2, x1:x2]
            data_dict[label_key] = crop_label_im

        return data_dict


class RemoveZeroSlicesd(Transform):
    """Removes all zero slices from the mask & corresponding slices in image."""

    # TODO: This is a wrong implementation but mostly gives the correct output, As rarely we see masks with zero masks in between.
    def __init__(self, add_pad: int = 0):
        """_summary_

        Args:
            add_pad (int, optional): Minimum padding in z direction, By default will remove all slices before & after . Defaults to 0.
        """
        self.add_pad = add_pad

    def __call__(self, data_dict: dict):
        scan_im = data_dict["image"]
        mask_im = data_dict["mask"]
        try:
            fil_slices = np.unique(np.nonzero(mask_im)[0])
            if np.min(fil_slices) - self.add_pad > 0:
                fil_slices = np.concatenate(
                    (
                        [
                            z
                            for z in range(
                                np.min(fil_slices) - self.add_pad, np.min(fil_slices)
                            )
                        ],
                        fil_slices,
                    )
                )

            if np.max(fil_slices) + self.add_pad + 1 < mask_im.shape[0]:
                fil_slices = np.concatenate(
                    (
                        fil_slices,
                        [
                            z
                            for z in range(
                                np.max(fil_slices) + 1,
                                np.max(fil_slices) + 1 + self.add_pad,
                            )
                        ],
                    )
                )

            # breakpoint()
            scan_im = scan_im[fil_slices]
            mask_im = mask_im[fil_slices]
            data_dict["image"] = scan_im
            data_dict["mask"] = mask_im
        except Exception as e:
            # print(f"Scan im shape: {scan_im.shape} {data_dict.keys()}")
            logger.debug(
                f"Exception: {e},\nScan im shape: {scan_im.shape} {data_dict.keys()} "
            )

        return data_dict


class ChangeView(Transform):
    """Change the view of a axial image to coronal or sagittal."""

    def __init__(self, mode="axial"):
        assert mode in [
            "axial",
            "coronal",
            "sagittal",
        ], f"mode: {mode} not supported. Supports ['axial', 'coronal', 'sagittal']."
        self.mode = mode

    def __call__(self, x):
        if len(x.shape) == 3:  # 3d volume
            base_index = 0
        elif len(x.shape) == 4:  # 3d volume with chns
            base_index = 1
        else:
            raise NotImplementedError(
                "Only supports 3D volume with or without channels"
            )

        if self.mode == "axial":
            return x
        elif self.mode == "coronal":
            return np.rollaxis(x, 1 + base_index, 0 + base_index)
        elif self.mode == "sagittal":
            return np.rollaxis(x, 2 + base_index, 0 + base_index)


class MIP(Transform):
    def __init__(self, mode="max", num_slices=5):
        self.mode = mode
        self.num_slices = num_slices

    def __call__(self, x):
        if len(x.shape) == 3:
            return create_MIP_numpy(x, num_slices=self.num_slices, mode=self.mode)
        else:
            return np.expand_dims(
                create_MIP_numpy(
                    np.squeeze(x, axis=0), num_slices=self.num_slices, mode=self.mode
                ),
                axis=0,
            )


class ApplyWindows(Transform):
    def __init__(self, keys: List[str], window: str):
        self.keys = keys
        assert (
            window in WINDOWS.keys()
        ), f"{window} not in supported windows. {list(WINDOWS.keys())}"
        self.window = WINDOWS[window]

    @staticmethod
    def window_generator(window_width, window_level):
        """Return CT window transform for given width and level."""
        low = window_level - window_width / 2
        high = window_level + window_width / 2

        def window_fn(img):
            img = (img - low) / (high - low)
            img = torch.clip(img, 0, 1)
            return img

        return window_fn

    def __call__(self, data: Dict):
        for key in self.keys:
            window_func = self.window_generator(self.window[1], self.window[0])
            data[key] = window_func(data[key])
        return data


class ApplyWindowsChannelWise(Transform):
    def __init__(self, keys: List[str], windows: List[str]):
        assert (
            len(set(windows).difference(list(WINDOWS.keys()))) == 0
        ), f"Wrong windows. Available options {list(WINDOWS.keys())}"
        self.keys = keys
        self.windows = [WINDOWS[window] for window in windows]

    def __call__(self, data: Dict):
        for key in self.keys:
            data[key] = torch.stack(
                [
                    ApplyWindows.window_generator(window[1], window[0])(data[key])
                    for window in self.windows
                ],
                0,
            )
        return data


class UpsampleZDims(Transform):
    def __init__(self, image_key: str, mask_key: str, anti_aliasing: bool = True):
        self.anti_aliasing = anti_aliasing
        self.image_key = image_key
        self.mask_key = mask_key

    def __call__(self, data: Dict):
        ratio = int(
            (data[self.image_key].shape[2] + data[self.image_key].shape[3])
            / (2 * data[self.image_key].shape[1])
        )
        ratio = np.min([np.max([1, ratio]), 3])
        data[self.mask_key + "_org"] = data[self.mask_key]
        data["z_ratio"] = ratio
        data[self.image_key] = rescale(
            data[self.image_key], scale=(1, ratio, 1, 1), anti_aliasing=True
        )
        data[self.mask_key] = (
            rescale(data[self.mask_key], scale=(1, ratio, 1, 1), anti_aliasing=True) > 0
        ).astype(np.int16)
        return data


class UpsampleDivisiblePadd(DivisiblePadd):
    def __init__(self, mask_key: str, image_key: str, **kwargs):
        super().__init__(
            keys=[mask_key, image_key, mask_key + "_org"],
            allow_missing_keys=True,
            **kwargs,
        )


class ResizeImage(Transform):
    """
    Resize the images and/or labels in the input data dictionary to a specified size.

    Args:
        size: size to which the images/labels will be resized. Can be a single integer or a sequence of integers
            representing height and width. If a sequence is passed, the transform will assume that the first two
            elements of the sequence represent height and width respectively.
        keys: List of strings representing the keys in the input data dictionary that correspond to the images/labels
            that will be resized.
        interpolation: Dictionary with keys corresponding to the `keys` input and values being strings representing
            the interpolation mode to be used during resizing. Valid options are `nearest`, `bilinear`, `bicubic`,
            `box`, `hamming`, `lanczos`, and `none`. If `None` is passed, `bilinear` interpolation is used by default.

    Returns:
        The modified input data dictionary with the resized images/labels.
    """

    def __init__(
        self,
        size: Union[Sequence[int], int],
        keys: List[str] = ["image", "label"],
        interpolation: Dict[str, str] = None,
    ):
        if isinstance(size, Sequence):
            size = tuple(size)
        self.resize = {}
        self.keys = keys
        if interpolation is None:
            interpolation = {key: "bilinear" for key in self.keys}
        for key in self.keys:
            self.resize[key] = transforms.Resize(
                size=size,
                interpolation=InterpolationMode(interpolation[key]),
                antialias=False,
            )

    def __call__(self, data: Dict) -> Dict:
        start_time = time.time()
        for key in self.keys:
            if isinstance(data[key], torch.Tensor):
                data[key] = self.resize[key](data[key])
        # logger.info(f"ResizeImage: {time.time() - start_time:.4f} s")
        return data


class ExpandDim(Transform):
    """
    Expands the dimensions of the specified keys to 4 if their dimensions are less than 4.
    If a key already has 4 dimensions, the transform does not change its dimension.

    Args:
        keys: list of keys to be transformed. Default value is ["image", "label"].
    """

    def __init__(self, keys: List[str] = ["image", "label"]):
        self.keys = keys

    def __call__(self, data: Dict) -> Dict:
        """
        Apply the transform to the input dictionary.

        Args:
            data: a dictionary containing the input data.

        Returns:
            a dictionary containing the transformed data.
        """
        for key in self.keys:
            if data[key].ndim < 4:
                data[key] = data[key].unsqueeze(0)
        for key in self.keys:
            assert (
                data[key].ndim == 4
            ), f"{key} should have 4 dimensions, not {data[key].ndim}"

            # logger.debug(f"{key} : {data[key].shape}")

        return data


class RandomAffine(Transform):
    """
    Randomly applies an affine transformation with rotation to the image and label data.

    Args:
        keys (List[str]): Keys to pick data for transformation.
        degrees (Sequence[float]): Range of rotation degrees. Default (-20,20).
        scale (Sequence[float]): Range of scale factors for zooming. Default (0.8, 1.2).
        prob (float): Probability of applying the transformation. Default 0.1.
    """

    def __init__(
        self,
        keys: List[str],
        degrees: Sequence[float] = (-20, 20),
        scale: Sequence[float] = (0.8, 1.2),
        prob: float = 0.1,
    ) -> None:
        self.keys = keys
        self.prob = prob
        self.rotate = transforms.RandomAffine(
            scale=tuple(scale), degrees=tuple(degrees)
        )

    def __call__(self, data: Dict):
        """
        Applies a random affine transformation with rotation to the image and label data.

        Args:
            data (Dict): A dictionary containing the data to be transformed.

        Returns:
            Dict: The transformed data.
        """
        if random.random() < self.prob:
            # Save the current state of the random number generator to ensure
            # that the same transform is applied to all keys
            state = torch.get_rng_state()
            for key in self.keys:
                torch.set_rng_state(state)
                data[key] = self.rotate(data[key])
        return data


class Rescale3D(Transform):
    """
    A Monai transform for downscaling 3D volumes using interpolation.

    """

    def __init__(
        self,
        keys: List[str],
        scale_factor_z: float = 1.0,
        scale_factor_y: float = 1.0,
        scale_factor_x: float = 1.0,
    ) -> None:
        """
        Initializes a new instance of the DownScale3D class.

        Args:
            keys (List[str]): A list of keys in the input data dictionary corresponding to the volumes to be downsampled.
            downscale_factor_z (int): The downsampling factor in the z-dimension. Default is 1.
            downscale_factor_y (int): The downsampling factor in the y-dimension. Default is 1.
            downscale_factor_x (int): The downsampling factor in the x-dimension. Default is 1.

        """
        self.keys = keys
        self.scale_factor = (scale_factor_z, scale_factor_y, scale_factor_x)
        self.target_ndim = 5

    def __call__(self, data: Dict) -> Dict:
        """
        Downsamples the volumes in the input data dictionary.

        Args:
            data (Dict): The input data dictionary.

        Returns:
            Dict: The input data dictionary with the specified keys downsampled.

        """
        for key in self.keys:
            num_new_dims = (
                self.target_ndim - data[key].dim()
            )  # calculate number of new dimensions needed
            new_shape = (1,) * num_new_dims + data[key].shape  # create new shape tuple
            data[key] = data[key].reshape(*new_shape)  # reshape tensor with new shape

            data[key] = F.interpolate(
                data[key],
                scale_factor=self.scale_factor,
                mode="trilinear",
                antialias=True,
            )

            # Squeeze the tensor along the first dimension until the desired number of dimensions is reached
            for i in range(num_new_dims):
                data[key] = torch.squeeze(data[key], dim=0)

        return data


class FixedZDim(Transform):
    """
    A Monai transform for cropping or padding 3D volumes to a fixed z-dimension.
    """

    def __init__(
        self,
        keys: List[str] = ["image"],
        target_z: int = 1,
        pad_value: str = "min",
        random_center_crop: bool = True,
    ) -> None:
        self.keys = keys
        self.target_z = target_z
        self.pad_value = pad_value
        self.random_center_crop = random_center_crop

    def __call__(self, data: Dict) -> Dict:
        # data_shapes = [data[key].shape[-3:] for key in self.keys]
        # assert len(set(data_shapes)) == 1, "All volumes must have the same shape"

        key = self.keys[0]
        image = data[key]
        # If size of the image is greater than target_z, crop the image
        if image.ndim == 3:
            image_z = image.shape[-3]
        if image.ndim == 1:
            image_z = image.shape[0]

        if image_z > self.target_z:
            if self.random_center_crop:
                z_start = random.randint(0, image_z - self.target_z)
            else:
                z_start = 0

            z_end = z_start + self.target_z

            for key in self.keys:
                data[key] = data[key][z_start:z_end]
            return data

        # If size of the image is less than target_z, pad the image
        elif image_z < self.target_z:
            for key in self.keys:
                image = data[key]
                if self.pad_value == "min":
                    pad = image.min()
                else:
                    pad = self.pad_value
                if image.ndim == 3:
                    data[key] = F.pad(
                        image,
                        (0, 0, 0, 0, 0, self.target_z - image_z),
                        mode="constant",
                        value=pad,
                    )
                if image.ndim == 1:
                    data[key] = F.pad(
                        image,
                        (0, self.target_z - image_z),
                        mode="constant",
                        value=pad,
                    )

        return data


class LoadDcmData(Transform):
    """
    A Monai transform for loading DICOM data into a PyTorch Tensor.

    """

    def __init__(self, keys: List[str]) -> None:
        """
        Initializes a new instance of the LoadDcmData class.
        Args:
            keys (List[str]): A list of keys in the input data dictionary that contain DICOM file paths.
        """
        self.keys = keys

    def __call__(self, data: Dict) -> Dict:
        """
        Loads DICOM data into PyTorch Tensors.

        Args:
            data (Dict): The input data dictionary.

        Returns:
            Dict: The input data dictionary with the specified keys replaced by PyTorch Tensors.
        """

        start_time = time.time()
        for key in self.keys:
            if isinstance(data[key], list):
                dcm_paths = data[key]
                arr_list = [dcmread(dcm_path).pixel_array for dcm_path in dcm_paths]
                arr = np.array(arr_list)
                data[key] = torch.Tensor(arr)

        # logger.info(f"Loading DICOM data took {time.time() - start_time} seconds")
        return data


class LoadSitkData(Transform):
    """
    A Monai transform for loading sitk data into a PyTorch Tensor.

    """

    def __init__(self, keys: List[str]) -> None:
        """
        Initializes a new instance of the LoadSitkData class.
        Args:
            keys (List[str]): A list of keys in the input data dictionary that contain sitk file paths.
        """
        self.keys = keys

    def __call__(self, data: Dict) -> Dict:
        """
        Loads Sitk data into PyTorch Tensors.

        Args:
            data (Dict): The input data dictionary.

        Returns:
            Dict: The input data dictionary with the specified keys replaced by PyTorch Tensors.
        """

        if "lower" in data.keys():
            lower = data["lower"]
        else:
            lower = 0
        if "upper" in data.keys():
            upper = data["upper"]
        else:
            upper = None
        if "invert_ct" in data.keys():
            invert_ct = data["invert_ct"]
        else:
            invert_ct = False

        start_time = time.time()
        for key in self.keys:
            filepath = data[key]
            arr = sitk.GetArrayFromImage(sitk.ReadImage(filepath))
            arr = arr[lower:upper]
            if invert_ct:
                arr = np.flip(arr, axis=0)
            data[key] = torch.Tensor(arr.copy())
        # logger.info(f"Loading sitk data took {time.time() - start_time} seconds")
        return data


class LoadHDF5Data(Transform):
    """
    A Monai transform for loading HDF5 data into a PyTorch Tensor.

    """

    def __init__(self, keys: List[str], to_tensor=True) -> None:
        """
        Initializes a new instance of the LoadHDF5Data class.
        Args:
            keys (List[str]): A list of keys in the input data dictionary that contain sitk file paths.
        """
        self.keys = keys
        self.to_tensor = to_tensor

    def __call__(self, data: Dict) -> Dict:
        """
        Loads HDF5 data into PyTorch Tensors.

        Args:
            data (Dict): The input data dictionary.

        Returns:
            Dict: The input data dictionary with the specified keys replaced by PyTorch Tensors.
        """
        if "lower" in data.keys():
            lower = data["lower"]
        else:
            lower = 0
        if "upper" in data.keys():
            upper = data["upper"]
        else:
            upper = None
        if "invert_ct" in data.keys():
            invert_ct = data["invert_ct"]
        else:
            invert_ct = False

        data = deepcopy(data)

        # start_time = time.time()
        try:
            f2 = h5py.File(data["datapath"], "r")
        except Exception as e:
            return data
            # logger.debug(e)
            # logger.debug(data["datapath"])

        for key in self.keys:
            try:
                image = f2[key]
                arr = image[lower:upper]
                if invert_ct:
                    arr = np.flip(arr, axis=0)
                if self.to_tensor:
                    data[key] = torch.Tensor(arr.copy().astype("float32"))
                else:
                    data[key] = arr
            except Exception as e:
                continue
                # logger.debug(e)
                # logger.debug(data["datapath"])

        # logger.info(f"Loading hdf5 data took {time.time() - start_time} seconds")
        return data


class LoadSafeTensor(Transform):
    """
    A Monai transform for loading Safetensor data to tensor.

    """

    def __init__(self, image_key: str,) -> None:
        self.image_key = image_key

    def __call__(self, data: Dict) -> Dict:
        if data["data_type"] == 0 : 
            return data
        data = deepcopy(data)

        # start_time = time.time()
        with safe_open(data["datapath"], framework="pt") as f:
            tensor_slice = f.get_slice("images")
            image_tensor = tensor_slice[
                data["start_idx"] : data["end_idx"], :, :, :
            ].to(dtype=torch.float32)
            tensor_labels = f.get_slice("labels")
            label_tensor = tensor_labels[data["start_idx"] : data["end_idx"]]

        data[self.image_key] = image_tensor.squeeze(0)
        # data["label"] = label_tensor
        del data["start_idx"]
        del data["end_idx"]
        # logger.info(f"Loading safetensor took {time.time() - start_time} seconds")
        return data
    

class LoadSafeTensor_FPR_SL(Transform):
    """
    A Monai transform for loading Safetensor data to tensor.

    """

    def __init__(self, image_key: str) -> None:
        self.image_key = image_key

    def __call__(self, data: Dict) -> Dict:
        if data["data_type"] == 1 : 
            return data
        data = deepcopy(data)

        bbox = data["bbox"]

        # start_time = time.time()
        with safe_open(data["datapath"], framework="pt") as f:
            tensor_slice = f.get_slice("image")
            image_tensor = tensor_slice[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]].to(dtype=torch.float32)

        data[self.image_key] = image_tensor

        del data["bbox"]
        # logger.info(f"Loading safetensor took {time.time() - start_time} seconds")
        return data
    

class ToTensor(Transform):
    """
    A Monai transform to convert data into a PyTorch Tensor.
    """

    def __init__(self, keys: List[str]) -> None:
        self.keys = keys

    def __call__(self, data: Dict) -> Dict:
        from copy import deepcopy

        data = deepcopy(data)
        for key in self.keys:
            data[key] = torch.Tensor(data[key]).to(torch.int64)
        return data


class CropHeadCT(Transform):
    def __init__(
        self,
        image_key: str,
        crop_key: str,
        margin: Tuple[int, int] = (0.1, 0.1),
    ) -> None:
        self.image_key = image_key
        self.crop_key = crop_key
        self.margin = margin

    def __call__(self, data: Dict) -> Dict:
        if self.crop_key not in data.keys():
            data[self.crop_key] = None

        if data[self.crop_key] is None:
            windowed_scan = ApplyWindows.window_generator(2400, 1800)(
                data[self.image_key]
            )
            crop_limits = [
                [0, -1],
                [0, -1],
                [0, -1],
            ]
            for i in [1, 2]:
                line = np.sum(windowed_scan.numpy(), axis=tuple({0, 1, 2} - {i}))
                line = line > 0.2

                ends = np.concatenate([[1], np.diff(line), [1]])
                ends = np.where(ends)[0]

                lengths = np.diff(ends)
                mid_idxs = (ends[1:] + ends[:-1]) // 2
                is_pos = line[mid_idxs]

                largest = np.argmax(lengths * is_pos)
                left, right = ends[largest], ends[largest + 1]

                left = max(left - round(self.margin[0] * len(line)), 0)
                right = min(right + round(self.margin[1] * len(line)), len(line))

                crop_limits[i] = [left, right]

            data[self.crop_key] = crop_limits

        return data

        # data[self.image_key] = data[self.image_key][
        #     :,
        #     crop_limits[0][0] : crop_limits[0][1],
        #     crop_limits[1][0] : crop_limits[1][1],
        # ]
        # if self.mask_keys is not None:
        #     for key in self.mask_keys:
        #         data[key] = data[key][
        #             :,
        #             crop_limits[0][0] : crop_limits[0][1],
        #             crop_limits[1][0] : crop_limits[1][1],
        #         ]

        # return data


# class CTCrop(Transform):
#     def __init__(
#         self,
#         image_key: str = "image",
#         mask_keys: Optional[Union[List[str], str]] = None,
#         body_part: str = "lung",
#         margin: Union[int, ITKDIM3D, ITKDIM2D] = ITKDIM3D(x=0, y=0, z=0),
#         random_prob: float = 0.5,
#     ) -> None:
#         self.body_crop = BodyCrop(body_part, margin, "cpu")
#         self.image_key = image_key
#         self.mask_keys = mask_keys
#         if self.mask_keys :
#             if isinstance(self.mask_keys, str):
#                 self.mask_keys = [self.mask_keys]
#             self.keys = [self.image_key] + self.mask_keys

#         else:
#             self.keys = [self.image_key]
#         self.margin = margin
#         self.random_crop_prob = random_prob

#     def __call__(self, data: Dict):
#         if random.random() > self.random_crop_prob:
#             return data
#         if isinstance(self.margin, int):
#             if data[self.image_key].ndim == 2:
#                 self.margin = ITKDIM2D(x=self.margin, y=self.margin)
#             else:
#                 self.margin = ITKDIM3D(x=self.margin, y=self.margin, z=self.margin)

#         assert (
#             data[self.image_key].ndim == 3
#         ), "To use this transform input data should be 3D Tensor "
#         self.body_crop.get_body_crop_bbox(data[self.image_key].numpy())
#         bbox = self.body_crop.bbox
#         bbox.pad(pad_dim=self.margin)
#         # start_time = time.time()
#         for key in self.keys:
#             data[key] = data[key][
#                 bbox.z1 : bbox.z2, bbox.y1 : bbox.y2, bbox.x1 : bbox.x2
#             ]
#         # logger.info(f"CT Crop took {time.time() - start_time} seconds")
#         return data


class ResampleSlice(Transform):
    """
    ResampleSlice is a transform class used for resampling slices of data.

    Args:
        keys (List[str]): List of keys representing the data to be resampled.
        spatial_z_threshold (int, optional): The threshold for resampling along the z-axis. It should be equal to Resize3D target z-dimension. Defaults to 160.
        random_prob (float, optional): The probability of applying the resampling. Defaults to 0.5.
    """

    def __init__(
        self, keys: List[str], spatial_z_threshold: int = 160, random_prob: float = 0.5
    ) -> None:
        self.keys = keys
        self.spatial_z_threshold = spatial_z_threshold
        self.random_sample_prob = random_prob

    def __call__(self, data: Dict):
        if random.random() > self.random_sample_prob:
            return data

        for keys in self.keys:
            resample = round(data[keys].shape[0] / self.spatial_z_threshold)
            if resample > 1:
                if random.random() > 0.5:
                    data[keys] = data[keys][::resample]
                else:
                    data[keys] = data[keys][1::resample]

        return data


class RandomGaussianBlur(Transform):
    def __init__(
        self,
        keys: List[str],
        kernel_size: int,
        sigma: Union[float, Tuple[float]],
        prob: float = 0.5,
    ) -> None:
        self.keys = keys
        self.random_blur_prob = prob
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.transform = transforms.GaussianBlur(
            kernel_size=self.kernel_size, sigma=self.sigma
        )

    def __call__(self, data: Dict):
        if random.random() > self.random_blur_prob:
            return data

        for key in self.keys:
            img = data[key]
            data[key] = self.transform(img)
        return data


class RandomZPad(Transform):
    """
    A Monai transform to padding 3D volumes if image z-dimension is less than target z-dimension.
    target z-dimension should be equal to Resize3D target z-dimension.
    It should be used before Resize3D.
    This is used as a augmentation strategies hence we using probability to apply this transform.
    """

    def __init__(
        self,
        keys: List[str] = ["image"],
        target_z: int = 1,
        pad_value: str = "min",
        prob: float = 0.5,
    ) -> None:
        self.keys = keys
        self.target_z = target_z
        self.pad_value = pad_value
        self.random_pad_prob = prob

    def __call__(self, data: Dict) -> Dict:
        if random.random() > self.random_pad_prob:
            return data

        data_shapes = [data[key].shape[-3:] for key in self.keys]
        assert len(set(data_shapes)) == 1, "All volumes must have the same shape"

        for key in self.keys:
            image = data[key]

            # If size of the image is less than target_z, pad the image
            if image.shape[-3] < self.target_z:
                if self.pad_value == "min":
                    pad = image.min()
                else:
                    pad = 0
                data[key] = F.pad(
                    image,
                    (0, 0, 0, 0, 0, self.target_z - image.shape[-3]),
                    mode="constant",
                    value=pad,
                )

        return data


class RandomHorizontalFlip(Transform):
    def __init__(self, keys: List[str] = ["image"], prob: float = 0.5) -> None:
        self.keys = keys
        self.prob = prob
        self.flip_h = transforms.RandomHorizontalFlip(p=1)

    def __call__(self, data: Dict) -> Dict:
        if random.random() < self.prob:
            state = torch.get_rng_state()
            for key in self.keys:
                torch.set_rng_state(state)
                data[key] = self.flip_h(data[key])
        return data


class ApplyCrop(Transform):
    def __init__(self, keys: List[str] = ["image"], crop_key: str = "crop") -> None:
        self.keys = keys
        self.crop_key = crop_key

    def __call__(self, data: Dict) -> Dict:
        crop = data[self.crop_key]
        if crop is not None:
            for key in self.keys:
                image = data[key]
                data[key] = image[:, crop[1][0] : crop[1][1], crop[2][0] : crop[2][1]]
        return data
    
class ApplyCropBBox(Transform):
    def __init__(self, keys: List[str] = ["image"], crop_key: str = "bbox") -> None:
        self.keys = keys
        self.crop_key = crop_key

    def __call__(self, data: Dict) -> Dict:
        crop = data[self.crop_key]
        if crop is not None:
            for key in self.keys:
                image = data[key]
                data[key] = image[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]
        return data


class IgnoreIndexMask(Transform):
    def __init__(self, ignore_index: int, key_shape: Dict[str, List]) -> None:
        self.key_shape = key_shape
        self.ignore_index = ignore_index

    def __call__(self, data: Dict) -> Dict:
        for key, value in self.key_shape.items():
            if isinstance(data[key], int):
                if data[key] == self.ignore_index:
                    data[key] = torch.ones(tuple(value)) * -100
        return data


class BinarizeSegMask(Transform):
    def __init__(self, keys: List[str]) -> None:
        self.keys = keys

    def __call__(self, data: Dict) -> Dict:
        for key in self.keys:
            # if len(torch.unique(data[key])) > 2 :
            im = data[key]
            im[im >= 1] = 1
            data[key] = im
        return data


# noise filter estimated using barlett's method for hct scans.
NOISE_FILTER = [
    [0.0302, 0.0137, -0.0062, -0.0295, -0.0062, 0.0137, 0.0302],
    [0.0137, 0.0121, 0.0294, 0.0654, 0.0294, 0.0121, 0.0137],
    [-0.0062, 0.0294, 0.1325, 0.3059, 0.1325, 0.0294, -0.0062],
    [-0.0295, 0.0654, 0.3059, 0.7224, 0.3059, 0.0654, -0.0295],
    [-0.0062, 0.0294, 0.1325, 0.3059, 0.1325, 0.0294, -0.0062],
    [0.0137, 0.0121, 0.0294, 0.0654, 0.0294, 0.0121, 0.0137],
    [0.0302, 0.0137, -0.0062, -0.0295, -0.0062, 0.0137, 0.0302],
]


class RandomNoise(Transform):
    """Add random noise to the image.

    Parameters
    ----------
    intensity: float
        Float in [0, 1]. Peak standard deviation for the gaussian noise.
    mode: str
        One of ('white', 'crop'). White is usual white noise. Crop adds noise
        only in regions which are not completely black or white.
    """

    def __init__(self, keys, intensity=0.25, mode="hct_thin", prob=0.5):
        self.keys = keys
        self.intensity = intensity
        self.mode = mode
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if random.random() > self.prob:
            return data

        rng = np.random.RandomState(random.randint(0, 200))
        scale = rng.uniform(0, self.intensity)

        for key in self.keys:
            img = data[key]  # (32,512,512)
            if self.mode == "white":
                noise = np.random.normal(loc=0, scale=scale, size=img.shape)
                data[key] = img + noise
            elif self.mode == "crop":
                crop = (img > 0.05) & (img < 0.95)
                noise = np.random.normal(loc=0, scale=scale * crop, size=img.shape[-2:])
                if img.ndim == 3:
                    noise = torch.Tensor(noise).unsqueeze(0)
                if img.ndim == 2:
                    noise = torch.Tensor(noise)
                data[key] = img + noise
            elif self.mode == "hct_thin":
                noise = np.random.normal(loc=0, scale=scale, size=img.shape[-2:])
                noise = ndi.filters.convolve(noise, NOISE_FILTER)
                if img.ndim == 3:
                    noise = torch.Tensor(noise).unsqueeze(0)
                if img.ndim == 2:
                    noise = torch.Tensor(noise)
                data[key] = img + noise
            else:
                raise NotImplementedError(f"{self.mode} not implemented")
        return data


def _sharpen_channel(img, intensity=1, kernel_size=3):
    arr = img.clone().numpy()
    """intensity in [-1, 1]."""
    n = kernel_size
    w = n ** (2 * intensity)
    center = (n - 1) // 2
    wts = np.full((n, n), (1 - w) / (n**2 - 1))
    wts[center, center] = w

    assert np.allclose(wts.sum(), 1), f"{wts.sum()} != 1"

    sharp_img = ndi.filters.convolve(arr, wts)
    return torch.Tensor(sharp_img)


class RandomSmoothSharpen(Transform):
    """Randomly smoothen or sharpen the image.

    Parameters
    ----------
    intensity: float
        Float in [0, 1]. Value of 0 would mean no augmentation. Value of 1
        would randomly select kernel b/w maximum smoothing and maximum
        sharpening.
    """

    def __init__(self, keys, intensity=0.5, kernel_size=3, prob=0.5):
        self.keys = keys
        self.kernel_size = kernel_size
        self.intensity = intensity
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if random.random() > self.prob:
            return data

        for key in self.keys:
            img = data[key]
            if img.ndim == 2:
                data[key] = _sharpen_channel(img, self.intensity, self.kernel_size)
            else:
                new_img = img.clone()
                for i in range(img.shape[0]):
                    new_img[i, :, :] = _sharpen_channel(
                        new_img[i, :, :], self.intensity, self.kernel_size
                    )
                data[key] = new_img
        return data


class Cutout(Transform):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (float): The length (in proportion of image height) of each
            square patch.
    """

    def __init__(self, keys, n_holes=20, length=0.08):
        self.keys = keys
        self.n_holes = n_holes
        self.length = length

    def __call__(self, data: Dict) -> Dict:
        seed = 20

        for key in self.keys:
            img = data[key]
            h, w = img.shape[-2:]
            rng = np.random.RandomState(seed)
            cutout_length = round(self.length * rng.uniform() * h)
            num_holes = round(self.n_holes * rng.uniform())
            mask = torch.ones((h, w), dtype=torch.float32)

            if img.ndim == 2:
                mask = torch.ones((h, w), dtype=torch.float32)
                for n in range(num_holes):
                    y = np.random.randint(h)
                    x = np.random.randint(w)

                    y1 = np.clip(y - cutout_length // 2, 0, h)
                    y2 = np.clip(y + cutout_length // 2, 0, h)
                    x1 = np.clip(x - cutout_length // 2, 0, w)
                    x2 = np.clip(x + cutout_length // 2, 0, w)

                    mask[y1:y2, x1:x2] = 0.0
            else:
                mask = torch.ones((img.shape[0], h, w), dtype=torch.float32)
                for z in range(img.shape[0]):
                    for n in range(num_holes):
                        y = np.random.randint(h)
                        x = np.random.randint(w)

                        y1 = np.clip(y - cutout_length // 2, 0, h)
                        y2 = np.clip(y + cutout_length // 2, 0, h)
                        x1 = np.clip(x - cutout_length // 2, 0, w)
                        x2 = np.clip(x + cutout_length // 2, 0, w)

                        mask[z, y1:y2, x1:x2] = 0.0

        return data


# class BoundaryExtract(Transform):
#     def __init__(self,
#                  keys: List[str],
#                  di: int ,
#                  er: int,
#                  ) -> None:
#         self.keys = keys
#         self.di = di
#         self.er = er
#         self.dilation_kernel_t = torch.ones((di,di,di)).unsqueeze(0).unsqueeze(0)
#         self.erosion_kernel_t = torch.ones((er,er,er)).unsqueeze(0).unsqueeze(0)

#     def __call__(self, data: Dict) -> Dict:
#         for key in self.keys:
#             mask = data[key]
#             if self.di > 0 :
#                 dilated_mask  = torch.clamp(torch.nn.functional.conv3d(mask, self.dilation_kernel_t, padding=(1,1,1)), 0, 1)
#                 data[key] = dilated_mask - mask

#             if self.er > 0 :
#                 eroded_mask = 1 - torch.clamp(torch.nn.functional.conv3d(1 - mask, self.erosion_kernel_t, padding=(1,1,1)), 0, 1)
#                 data[key] = mask - eroded_mask

#         return data


class ExtractICVBoundary(Transform):
    def __init__(
        self, keys: List[str], di: int = 3, er: int = 3, slc_range: List = [14, 28]
    ) -> None:
        self.keys = keys
        self.di = di
        self.er = er
        self.slc_range = slc_range

    def __call__(self, data: Dict) -> Dict:
        # start_time = time.time()
        for key in self.keys:
            mask = data[key]
            if self.er > 0:
                eroded_icv_mask_arr = np.zeros(mask.shape)
                for i in range(
                    self.slc_range[0], min(self.slc_range[1], mask.shape[0])
                ):
                    eroded_icv_mask_arr[i] = binary_erosion(mask[i], disk(self.er))

            if self.di > 0:
                dilated_icv_mask_arr = np.zeros(mask.shape)
                for i in range(
                    self.slc_range[0], min(self.slc_range[1], mask.shape[0])
                ):
                    dilated_icv_mask_arr[i] = binary_erosion(mask[i], disk(self.di))

            data[key] = dilated_icv_mask_arr - eroded_icv_mask_arr
        # logger.info(f"Extract ICV contour took {time.time() - start_time} seconds")
        return data


class GenerateICV(Transform):
    def __init__(self, image_key: str = "image") -> None:
        super().__init__()
        self.image_key = image_key

    def __call__(self, data: Dict) -> Dict:
        if "icv_mask" in data.keys():
            return data

        from qer.utils.preprocessing.windowing import window_generator

        custom_window = window_generator(900, 50)
        brain_window = window_generator(80, 40)

        arr = data[self.image_key]

        a = (custom_window(arr) < 0.2) * 1
        icv = np.zeros(arr.shape)
        b = (brain_window(arr) > 0.8) * 1
        for i in range(14, min(icv.shape[0], 28)):
            icv[i] = binary_fill_holes(b[i], structure=np.ones((3, 3))).astype(int)

        b[0:14] = 0
        d = icv - b
        f = d.copy()
        f[a == 1] = 0

        e = np.zeros(f.shape)
        for i in range(e.shape[0]):
            e[i] = erosion(f[i], disk(2))

        h = np.zeros(e.shape)
        for i in range(e.shape[0]):
            h[i] = get_largest_island(e[i])

        g = np.zeros(h.shape)
        for i in range(h.shape[0]):
            g[i] = dilation(h[i], disk(2))

        t = np.zeros(g.shape)
        for i in range(g.shape[0]):
            t[i] = closing(g[i], disk(2))

        data["icv_mask"] = t

        return data


class ExtractFalx(Transform):
    def __init__(
        self,
        key: str = "hemi_seperator_mask",
        icv_mask_key: str = "icv_mask",
        dilation_kernel_size: int = 15,
        slc_range: List[int] = [16, 28],
        output_key: str = "falx",
    ) -> None:
        self.key = key
        self.icv_mask_key = icv_mask_key
        self.dilation_kernel_size = dilation_kernel_size
        self.slc_range = slc_range
        self.output_key = output_key
        super().__init__()

    def __call__(self, data: Dict) -> Dict:
        hemi_seperator_mask = data[self.key]
        icv_mask = data[self.icv_mask_key]

        edge_mask = cv2.Canny(
            hemi_seperator_mask.astype(np.uint8), 0, 1
        )  # Using Canny edge detection
        dilated_edge_mask = cv2.dilate(
            edge_mask,
            np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8),
        )

        falx_region = dilated_edge_mask * icv_mask

        falx_region[0 : self.slc_range[0]] = 0
        if falx_region.shape[0] > self.slc_range[1]:
            falx_region[self.slc_range[1] :] = 0

        data[self.output_key] = falx_region
        del data[self.key]
        return data


import math


class CropTensor(Transform):
    def __init__(self, output_shape, key: str) -> None:
        self.output_shape = output_shape
        self.key = key

    def __call__(self, data: Any):
        input_tensor = data[self.key]
        D, H, W = input_tensor.shape
        d, h, w = self.output_shape

        start_d = math.floor((D - d) / 2)
        start_h = math.floor((H - h) / 2)
        start_w = math.floor((W - w) / 2)

        end_d = start_d + d
        end_h = start_h + h
        end_w = start_w + w

        cropped_tensor = input_tensor[start_d:end_d, start_h:end_h, start_w:end_w]

        data[self.key] = cropped_tensor

        return data