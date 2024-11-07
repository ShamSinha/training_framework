"""Lung nodule segmentation inference script."""
from typing import Dict, Union

import numpy as np
import torch
from loguru import logger
from monai.transforms import (
    AddChanneld,
    Compose,
    DivisiblePadd,
    EnsureTyped,
    FillHolesd,
    Invertd,
    KeepLargestConnectedComponentd,
    ScaleIntensityRanged,
    ThresholdIntensityd,
)
from monai.transforms.utils import allow_missing_keys_mode
from qct_utils.ct_schema import (
    ITKDIM,
    ITKDIMFLOAT,
    BigCtscan,
    Ctannotdim,
    Ctscan,
    CtscanAnnot,
    Extendbox,
)
from tqdm import tqdm

from src.common.nn_modules.nets.segmentation.Qnet import Qnet


def get_test_transforms():
    test_transforms = Compose(
        [
            AddChanneld(keys=["image", "roi_mask"]),
            ThresholdIntensityd(keys=["image"], above=False, threshold=600, cval=600),
            ThresholdIntensityd(keys=["image"], above=True, threshold=-1200, cval=-1200),
            ScaleIntensityRanged(keys=["image"], a_min=-1200, a_max=600),
            DivisiblePadd(keys=["image", "roi_mask"], k=16),
            EnsureTyped(keys=["image", "roi_mask"]),
        ]
    )
    post_transforms = Compose(
        [
            Invertd(
                keys="pred",  # invert the `pred` data field, also support multiple fields # noqa E501
                transform=test_transforms,
                orig_keys="roi_mask",  # get the previously applied pre_transforms information on the `img` data field, # noqa E501
                # then invert `pred` based on this information. we can use same info # noqa E501
                # for multiple fields, also support different orig_keys for different fields # noqa E501
            ),
            KeepLargestConnectedComponentd(
                keys=["pred"], applied_labels=[1], is_onehot=False, connectivity=2
            ),
            FillHolesd(keys=["pred"], applied_labels=[1], connectivity=3),
        ]
    )
    return test_transforms, post_transforms


def xyzcwhd2xyzxyz(bbox, size):
    """convert (x_center, y_center, z_center, w, h, d) to (x1, y1, z1, x2, y2, z2)"""
    x1, x2 = bbox.x_center - (bbox.w / 2), bbox.x_center + (bbox.w / 2)
    y1, y2 = bbox.y_center - (bbox.h / 2), bbox.y_center + (bbox.h / 2)
    z1, z2 = bbox.z_center - (bbox.d / 2), bbox.z_center + (bbox.d / 2)
    x1, y1, z1 = max(x1, 0), max(y1, 0), max(z1, 0)
    x2, y2, z2 = min(x2, size.x), min(y2, size.y), min(z2, size.z)
    return Extendbox(
        x1=round(x1, 0),
        y1=round(y1, 0),
        z1=round(z1, 0),
        x2=round(x2, 0),
        y2=round(y2, 0),
        z2=round(z2, 0),
    )


class NoduleSegmentation:
    def __init__(self, checkpoint_path: str, device="cuda", **kwargs):
        network = Qnet(num_classes=2)
        checkpoint_path = checkpoint_path
        self.model = self.load_model(
            ckpt_path=checkpoint_path, network=network, device=torch.device(device)
        )
        self.device = device
        self.test_transforms, self.post_transforms = get_test_transforms()

    @staticmethod
    def load_model(ckpt_path: str, network, device):
        state_dict = torch.load(ckpt_path)
        network.load_state_dict(state_dict)
        network.eval()
        network.to(device)
        logger.debug(f"Nodule Seg Model: Loaded weights from {ckpt_path}")
        return network

    def ctannotinput(
        self,
        scan: np.ndarray,
        annot: Union[CtscanAnnot, Ctannotdim],
        crop_margin: ITKDIM,
        roi_margin: ITKDIM,
        extendbox: bool = True,
    ) -> Dict[str, np.ndarray]:

        if isinstance(annot, CtscanAnnot):
            bbox = annot.annot.bbox
        elif isinstance(annot, Ctannotdim):
            bbox = annot.bbox
        else:
            raise NotImplementedError("Un supported annot")
        sz, sy, sx = scan.shape
        scan_dims = ITKDIM(x=sx, y=sy, z=sz)
        box = xyzcwhd2xyzxyz(bbox, scan_dims)

        z1 = max(0, int(box.z1) - crop_margin.z)
        z2 = min(sz, int(box.z2) + 1 + crop_margin.z)
        y1 = max(0, int(box.y1) - crop_margin.y)
        y2 = min(sy, int(box.y2) + 1 + crop_margin.y)
        x1 = max(0, int(box.x1) - crop_margin.x)
        x2 = min(sx, int(box.x2) + 1 + crop_margin.x)

        roi_mask = np.zeros_like(scan)
        roi_mask[
            max(0, int(box.z1) - roi_margin.z) : min(sz, int(box.z2) + 1 + roi_margin.z),
            max(0, int(box.y1) - roi_margin.y) : min(sy, int(box.y2) + 1 + roi_margin.y),
            max(0, int(box.x1) - roi_margin.x) : min(sx, int(box.x2) + 1 + roi_margin.x),
        ] = 1.0
        data_dict = {"image": scan[z1:z2, y1:y2, x1:x2], "roi_mask": roi_mask[z1:z2, y1:y2, x1:x2]}
        if extendbox:
            return data_dict, Extendbox(x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2)
        return data_dict

    def predict_numpy(self, data_dict: Dict, conf_thr: float = 0.5, apply_thr: bool = True):
        # data should be dict {"image": [...], "roi_mask": [....]}
        img_dict = self.test_transforms(data_dict)
        with torch.no_grad():
            out = self.model(
                img_dict["image"].unsqueeze(0).to(self.device).float(),
                img_dict["roi_mask"].unsqueeze(0).to(self.device).float(),
            )
            out_prob = torch.softmax(out, dim=1).cpu()
        cls_probs = out_prob[:, 1, ...]
        if apply_thr:
            cls_probs[cls_probs >= conf_thr] = 1
            cls_probs[cls_probs < conf_thr] = 0
            cls_probs = cls_probs.to(torch.int8)
        img_dict["pred"] = cls_probs
        with allow_missing_keys_mode(self.test_transforms):
            inverted_seg = self.post_transforms(img_dict)
            inverted_mask = inverted_seg["pred"][0]
        if isinstance(inverted_mask, torch.TensorType):
            inverted_mask = inverted_mask.cpu().numpy()
        return inverted_mask  # Returns DxHxW

    def predict_ctscan_annot(
        self,
        ndimg: np.ndarray,
        ctscanannot: CtscanAnnot,
        spacing: ITKDIMFLOAT,
        crop_margin: ITKDIM = ITKDIM(z=15, y=80, x=80),
        roi_margin: ITKDIM = ITKDIM(z=5, y=20, x=20),
        conf_thr: float = 0.5,
        volume_threshold: float = 0,
        axial_view_diameter: bool = True,
        store_numpy_masks: bool = True,
        store_volume_diameter: bool = True,
    ):
        annotdim: Ctannotdim = ctscanannot.annot
        if annotdim.spacing is None:
            annotdim.spacing = spacing
        data_dict, extendbox = self.ctannotinput(
            ndimg, ctscanannot, crop_margin=crop_margin, roi_margin=roi_margin
        )

        if np.prod(data_dict["image"].shape) == 0:
            logger.warning(
                f"removed {ctscanannot.Nodule_name} Since crop of nodule size is:\
                    {data_dict['image'].shape} Box conf is {annotdim.conf}"
            )
            return None

        out_mask: np.ndarray = self.predict_numpy(
            data_dict=data_dict, conf_thr=conf_thr, apply_thr=True
        )

        if out_mask.sum() >= volume_threshold:
            sz, sy, sx = ndimg.shape
            full_mask = np.zeros((sz, sy, sx)).astype(np.uint8)
            z1, z2, y1, y2, x1, x2 = (
                extendbox.z1,
                extendbox.z2,
                extendbox.y1,
                extendbox.y2,
                extendbox.x1,
                extendbox.x2,
            )
            full_mask[z1:z2, y1:y2, x1:x2] = out_mask
            annotdim.set_mask(
                mask=full_mask,
                store_mask=store_numpy_masks,
                axial_view_diameter=axial_view_diameter,
                store_volume_diameter=store_volume_diameter,
            )
        else:
            logger.warning(
                f"Not updating volume and nodule properties for {ctscanannot.Nodule_name} \
                Since predicted nodule +ve pixels are: {out_mask.sum()} Box conf is {annotdim.conf}"
            )
            return None
        return ctscanannot

    def predict_ctscan(
        self,
        ctscan: Ctscan,
        crop_margin: ITKDIM = ITKDIM(z=15, y=80, x=80),
        roi_margin: ITKDIM = ITKDIM(z=5, y=20, x=20),
        conf_thr: float = 0.5,
        volume_thr: float = 0,
        axial_view_diameter: bool = True,
        store_numpy_masks: bool = True,
    ):
        raw_nd_img = ctscan.get_numpy_array()
        predicted_annots = []
        for annot in tqdm(ctscan.Annot, total=len(ctscan.Annot), desc="Finetuning nodule masks"):
            predicted_annot = self.predict_ctscan_annot(
                raw_nd_img,
                annot,
                spacing=ctscan.Spacing,
                crop_margin=crop_margin,
                roi_margin=roi_margin,
                volume_threshold=volume_thr,
                conf_thr=conf_thr,
                axial_view_diameter=axial_view_diameter,
                store_numpy_masks=store_numpy_masks,
            )
            if predicted_annot is not None:
                predicted_annots.append(predicted_annot)
        ctscan.Annot = predicted_annots
        return ctscan

    def predict_bigctscan(
        self,
        big_ctscan: BigCtscan,
        predict_on_pred: bool = True,
        crop_margin: ITKDIM = ITKDIM(z=15, y=80, x=80),
        roi_margin: ITKDIM = ITKDIM(z=5, y=20, x=20),
        conf_thr: float = 0.5,
        axial_view_diameter: bool = True,
        store_numpy_masks: bool = True,
    ) -> BigCtscan:

        if predict_on_pred:
            big_ctscan.Pred.Scan = big_ctscan.Raw.Scan
            ctscan = big_ctscan.Pred
        else:
            ctscan = big_ctscan.Raw
        big_ctscan.Pred = self.predict_ctscan(
            ctscan=ctscan,
            crop_margin=crop_margin,
            roi_margin=roi_margin,
            conf_thr=conf_thr,
            axial_view_diameter=axial_view_diameter,
            store_numpy_masks=store_numpy_masks,
        )
        return big_ctscan
