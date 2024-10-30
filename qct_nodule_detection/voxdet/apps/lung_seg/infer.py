import numpy as np 
import torch 
import SimpleITK as sitk 
import fastcore.all as fc 
from tqdm import tqdm

from typing import Union, List, Optional 

from .resunet import UNet 
from .utils import preprocess, LungLabelsDS_inf, postrocessing, reshape_mask


class SegmentationModelBase:
    def __init__(self, checkpoint_path: str = None, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path 
        self.device = torch.device(device)

    @staticmethod
    def _load_model(ckpt_path: str, device):
        weights = torch.load(ckpt_path, map_location=torch.device("cpu"))
        state_dict = weights["state_dict"]
        num_classes = int(weights["num_classes"])
        model = UNet(
            n_classes=num_classes,
            padding=True,
            depth=5,
            up_mode="upsample",
            batch_norm=True,
            residual=False,
        )
        model.load_state_dict(state_dict)
        model.eval()

        return model.to(device=device)

    @torch.no_grad()
    def _predict(
        self,
        model,
        image: Union[sitk.Image, np.ndarray],
        batch_size=20,
        volume_postprocessing=True,
        noHU=False,
    ):
        inimg_raw = (
            sitk.GetArrayFromImage(image)
            if isinstance(image, sitk.Image)
            else image
        )
        directions = (
            np.asarray(image.GetDirection())
            if isinstance(image, sitk.Image)
            else None
        )  # skip direction check for ndarray
        if isinstance(image, sitk.Image) and len(directions) == 9:
            inimg_raw = np.flip(
                inimg_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0]
            )

        if not noHU:
            tvolslices, xnew_box = preprocess(
                inimg_raw, resolution=[256, 256]
            )
            tvolslices[tvolslices > 600] = 600
            tvolslices = np.divide((tvolslices + 1024), 1624)
        else:
            # support for non HU images. This is just a hack.
            # The models were not trained with this in mind
            tvolslices = skimage.color.rgb2gray(inimg_raw)
            tvolslices = skimage.transform.resize(tvolslices, [256, 256])
            tvolslices = np.asarray(
                [tvolslices * x for x in np.linspace(0.3, 2, 20)]
            )
            tvolslices[tvolslices > 1] = 1
            sanity = [
                (tvolslices[x] > 0.6).sum() > 25000
                for x in range(len(tvolslices))
            ]
            tvolslices = tvolslices[sanity]
        torch_ds_val = LungLabelsDS_inf(tvolslices)
        dataloader_val = torch.utils.data.DataLoader(
            torch_ds_val,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
        )

        timage_res = np.empty(
            (np.append(0, tvolslices[0].shape)), dtype=np.uint8
        )

        with torch.no_grad():
            for X in tqdm(dataloader_val, desc="predicting lung & lobe mask"):
                # print(X.shape)
                X = X.float().to(self.device)
                prediction = model(X)
                pls = (
                    torch.max(prediction, 1)[1]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                timage_res = np.vstack((timage_res, pls))
        # postprocessing includes removal of small connected
        # components, hole filling and mapping of small components to
        # neighbors
        if volume_postprocessing:
            outmask = postrocessing(timage_res)
        else:
            outmask = timage_res

        if noHU:
            outmask = skimage.transform.resize(
                outmask[np.argmax((outmask == 1).sum(axis=(1, 2)))],
                inimg_raw.shape[:2],
                order=0,
                anti_aliasing=False,
                preserve_range=True,
            )[None, :, :]
        else:
            outmask = np.asarray(
                [
                    reshape_mask(
                        outmask[i], xnew_box[i], inimg_raw.shape[1:]
                    )
                    for i in range(outmask.shape[0])
                ],
                dtype=np.uint8,
            )

        return outmask.astype(np.uint8)
        


class LungSegmentation(SegmentationModelBase):
    def __init__(self,  checkpoint_path: str,  device: str="cuda",  batch_size: int=20,  volume_postprocessing: bool=True, noHU: bool=False):
        super().__init__( checkpoint_path=checkpoint_path, device=device)
        fc.store_attr()
        self.model = None 
    
    @staticmethod
    def __get_seperate_lungs(lung_mask):
        ## left lung
        bw1 = np.zeros(lung_mask.shape)  # deepcopy(lung_mask)
        bw1[lung_mask == 1] = 1

        ## right lung
        bw2 = np.zeros(lung_mask.shape)  # deepcopy(lung_mask)
        # bw2[lung_mask == 1] = 0
        bw2[lung_mask == 2] = 1

        lung_mask[lung_mask >= 1] = 1

        bw1, bw2, lung_mask = (
            np.bool_(bw1),
            np.bool_(bw2),
            np.bool_(lung_mask),
        )

        return bw1, bw2, lung_mask

    def __extract_lungmask_with_cache(
        self,
        lung_scan: Union[sitk.Image, np.ndarray],
        lung_masks_save_dir: Optional[str],
        series_id: Optional[str],
    ):
        """
        If lung mask cache exists then loads left, right and full lung mask
        else processes the lung_scan to get the lung mask
        """

        # Check for cache
        if lung_masks_save_dir and series_id:
            lung_mask_path = os.path.join(
                lung_masks_save_dir, series_id + "_lung_mask.nii.gz"
            )
            if os.path.exists(lung_mask_path):
                logging.info(
                    f"Lung mask cache hit, found: lung_mask at {lung_mask_path}"
                )
                lung_mask_itk = sitk.ReadImage(lung_mask_path)
                return lung_mask_itk

        lung_mask = self._predict(
            model=self.model,
            image=lung_scan,
            batch_size=self.batch_size,
            volume_postprocessing=self.volume_postprocessing,
            noHU=self.noHU,
        )
        if lung_masks_save_dir and series_id:
            lung_mask_path = os.path.join(
                lung_masks_save_dir, series_id + "_lung_mask.nii.gz"
            )
            os.makedirs(os.path.dirname(lung_mask_path), exist_ok=True)
            lung_mask_itk = transfer_sitk_params(
                np.uint8(lung_mask), lung_scan
            )
            sitk.WriteImage(lung_mask_itk, lung_mask_path)
        return lung_mask

    
    def predict_lung_np(self, data: np.ndarray):
        if self.model is None:
            self.model = self._load_model(self.checkpoint_path, self.device)
        
        return self.__extract_lungmask_with_cache(
            data, lung_masks_save_dir=None, series_id=None
        )


