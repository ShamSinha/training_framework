from tqdm import tqdm
import pandas as pd
from torchvision.transforms import Compose
import albumentations as A
import glob
import numpy as np
import torch
import torch.nn as nn
import os
import cv2
from pathlib import Path
from cv2 import INTER_AREA
import torch.utils.data as data
from cv2 import imwrite
from numpy import uint8


def scale(arr: np.ndarray) -> np.ndarray:
    """
    scales all values of numpy array to [0,1]
    Args:
        arr: numpy array

    Returns:
        scaled array

    """
    # TODO assertion on type of numpy array ?

    eps = 1e-10
    if arr.dtype in [np.float64, np.float32, np.float16]:
        eps = np.finfo(arr.dtype).eps
    arr = arr - arr.min()
    arr = arr / (arr.max() + eps)
    return arr


class NDTransform(object):
    """Base class for all numpy based transforms.

    This class achieves the following:

    * Abstract the transform into
        * Getting parameters to apply which is only run only once per __call__.
        * Applying transform given parameters
    * Check arguments passed to a transforms for consistency

    Abstraction is especially useful when there is randomness involved with the
    transform. You don't want to have different transforms applied to different
    members of a data point.
    """

    def _argcheck(self, data):
        """Check data for arguments."""

        if isinstance(data, np.ndarray):
            assert data.ndim in {
                2,
                3,
            }, "Image should be a ndarray of shape H x W x C or H X W."
            if data.ndim == 3:
                assert (
                    data.shape[2] < data.shape[0]
                ), "Is your color axis the last? Roll axes using np.rollaxis."

            return data.shape[:2]
        elif isinstance(data, dict):
            for k, img in data.items():
                if isinstance(img, np.ndarray):
                    assert isinstance(k, str)

            shapes = {
                k: self._argcheck(img)
                for k, img in data.items()
                if isinstance(img, np.ndarray)
            }
            assert (
                len(set(shapes.values())) == 1
            ), "All member images must have same size. Instead got: {}".format(shapes)
            return set(shapes.values()).pop()
        else:
            raise TypeError("ndarray or dict of ndarray can only be passed")

    def _get_params(self, h, w, seed=None):
        """Get parameters of the transform to be applied for all member images.

        Implement this function if there are parameters to your transform which
        depend on the image size. Need not implement it if there are no such
        parameters.

        Parameters
        ----------
        h: int
            Height of the image. i.e, img.shape[0].
        w: int
            Width of the image. i.e, img.shape[1].

        Returns
        -------
        params: dict
            Parameters of the transform in a dict with string keys.
            e.g. {'angle': 30}
        """
        return {}

    def _transform(self, img, is_label, **kwargs):
        """Apply the transform on an image.

        Use the parameters returned by _get_params and apply the transform on
        img. Be wary if the image is label or not.

        Parameters
        ----------
        img: ndarray
            Image to be transformed. Can be a color (H X W X C) or
            gray (H X W)image.
        is_label: bool
            True if image is to be considered as label, else False.
        **kwargs
            kwargs will be the dict returned by get_params

        Return
        ------
        img_transformed: ndarray
            Transformed image.
        """
        raise NotImplementedError

    def __call__(self, data, seed=None):
        """
        Parameters
        ----------
        data: dict or ndarray
            Image ndarray or a dict of images. All ndarrays in the dict are
            considered as images and should be of same size. If key for a
            image in dict has string `target` in it somewhere, it is
            considered as a target segmentation map.
        """
        h, w = self._argcheck(data)
        params = self._get_params(h, w, seed=seed)

        if isinstance(data, np.ndarray):
            return self._transform(data, is_label=False, **params)
        else:
            data = data.copy()
            for k, img in data.items():
                if isinstance(img, np.ndarray):
                    if isinstance(k, str) and "target" in k:
                        is_label = True
                    else:
                        is_label = False

                    data[k] = self._transform(img.copy(), is_label, **params)
            return data


class ToTensor(NDTransform):
    """Convert ndarrays to tensors.

    Following are taken care of when converting to tensors:

    * Axes are swapped so that color axis is in front of rows and columns
    * A color axis is added in case of gray images
    * Target images are left alone and are directly converted
    * Label images is set to LongTensor by default as expected by torch's loss
      functions.

    Parameters
    ----------
    dtype: torch dtype
        If you want to convert all tensors to cuda, you can directly
        set dtype=torch.cuda.FloatTensor. This is for non label images
    dtype_label: torch dtype
        Same as above but for label images.
    """

    import torch

    def _transform(self, img, is_label):
        img = np.ascontiguousarray(img)
        if not is_label:
            # put it from HWC to CHW format
            if img.ndim == 3:
                img = np.rollaxis(img, 2, 0)
            elif img.ndim == 2:
                img = img.reshape((1,) + img.shape)
        else:
            if img.ndim == 3:  # making transforms work for multi mask models
                img = np.rollaxis(img, 2, 0)

        img = self.torch.from_numpy(img)

        if is_label:
            return img.long()
        else:
            return img.float()


def clip(arr: np.ndarray) -> np.ndarray:
    """
    Clips the array between 0 and 1
    Args:
        arr: numpy array

    Returns:
        clipped numpy array

    """
    arr = np.clip(arr, a_max=1.0, a_min=0)
    return arr


def rescale(arr: np.ndarray) -> np.ndarray:
    """
    scales all values of numpy array to [0,1]
    Args:
        arr: numpy array

    Returns:
        scaled array

    """
    arr = arr / 255
    return arr


class Resize_Scale:
    def __init__(self, im_size) -> None:
        self.im_size = im_size
        self.albumentation_transforms = A.ReplayCompose(
            [
                A.Resize(
                    self.im_size,
                    self.im_size,
                    interpolation=INTER_AREA,
                    always_apply=True,
                ),
            ]
        )

        self.transforms = Compose(
            [
                ToTensor(),
            ]
        )

    def __call__(self, original_image):
        try:
            tf_image = self.albumentation_transforms(image=original_image)
            original_image = rescale(tf_image["image"])
            original_image = scale(original_image)
            original_image = clip(original_image)
            augmented_image = original_image[np.newaxis, :, :]  # convert to rgb
            augmented_image = augmented_image.transpose(1, 2, 0)
            augmented_image = self.transforms(augmented_image)

        except Exception as e:
            print("\033[91m" + "applying transform error" + "\033[0m")
            print(e)
            return original_image

        return augmented_image


class dataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # filename = Path(str(df['dcm_path'][idx]).strip())
        filename = self.files[idx]
        # print(filename)
        # img = get_array_from_dicom(str(filename))
        img = cv2.imread(filename, 0)
        img = 255 * scale(img)
        img = self.transforms(img)
        return {"input": img, "filename": str(os.path.basename(filename)[:-4])}


if __name__ == "__main__":
    df = list(
        glob.glob(
            "/models_common_e2e/cxr_data/testing/tb_data/images/*.png",
            recursive=True,
        )
    )
    Data = dataset(df, Resize_Scale(1024))
    dataloader = data.DataLoader(
        Data,
        batch_size=32,
        num_workers=5,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
        prefetch_factor=2,
    )

    model = torch.jit.load(
        "/fast_data_2d_1/akshay/cxr_models/torchscript_traces/ensembled/gpu_models/v4_adult_tb_cuda.ts",
        # map_location="cuda:0",
    )
    model.eval()
    model = nn.DataParallel(model)
    new_df = {
        "filename": [],
        "adult_tb": [],
        "max_pixel_score": [],
        "mean_pixel_score": [],
        "min_pixel_score": [],
        "count_pixel_score": [],
        # "age": [],
    }
    with torch.no_grad():
        for i, dic in enumerate(tqdm(dataloader)):
            output = model(dic["input"].cuda())
            [new_df["filename"].append(x) for x in dic["filename"]]
            [new_df["adult_tb"].append(x.item()) for x in output[0]["tuberculosis"]]
            # [new_df["age"].append(x.item()) for x in output[2]["age"]]

            for j, seg_output in enumerate(output[1]["tuberculosis"]):
                seg = seg_output.cpu().numpy()
                new_df["max_pixel_score"].append(seg.max())
                new_df["mean_pixel_score"].append(seg.mean())
                new_df["min_pixel_score"].append(seg.min())
                new_df["count_pixel_score"].append(seg.sum())
                if output[0]["tuberculosis"][j] >= 0.5:
                    imwrite(
                        f'/fast_data_2d_3/akshay/TB_output/seg_pos_ens/{dic["filename"][j]+".png"}',
                        ((seg) * 255).astype(uint8),
                    )

            if i == 0:
                new_df = pd.DataFrame(new_df)
                new_df.to_csv(
                    "/fast_data_2d_3/akshay/TB_output/TB_final_model_results_deeplab_unet_unetpp.csv",
                    mode="a",
                    header=True,
                    index=False,
                )
                new_df = {
                    "filename": [],
                    "adult_tb": [],
                    "max_pixel_score": [],
                    "mean_pixel_score": [],
                    "min_pixel_score": [],
                    "count_pixel_score": [],
                    # "age": [],
                }
            elif i % 100 == 0:
                new_df = pd.DataFrame(new_df)
                new_df.to_csv(
                    "/fast_data_2d_3/akshay/TB_output/TB_final_model_results_deeplab_unet_unetpp.csv",
                    mode="a",
                    header=False,
                    index=False,
                )
                new_df = {
                    "filename": [],
                    "adult_tb": [],
                    "max_pixel_score": [],
                    "mean_pixel_score": [],
                    "min_pixel_score": [],
                    "count_pixel_score": [],
                    # "age": [],
                }

    new_df = pd.DataFrame(new_df)
    new_df.to_csv(
        "/fast_data_2d_3/akshay/TB_output/TB_final_model_results_deeplab_unet_unetpp.csv",
        mode="a",
        header=False,
        index=False,
    )
