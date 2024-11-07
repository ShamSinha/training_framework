from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from cxr_training.data.data_utils.utils import read_image_from_path
import glob

from cxr_training.data.transforms.transform_module import Base_transform


class Base_dataset(Dataset):
    """A torch dataset class which returns each item in the form of dict consisting of keys idx, input, target.
    Transforms are applied based dict keys.
    """

    def __init__(self, args, dataframe: pd.DataFrame, mode) -> None:
        super().__init__()
        self.mode = mode
        self.recipe = args.trainer.recipe
        self.args = args
        self.im_size = args.params.im_size
        self.mask_threshold = args.params.mask_threshold
        self.df = dataframe
        self.img_folder_path = args.files.img_folder_path
        self.annotation_path = args.files.annotation_path
        self.transforms = Base_transform(self.args, mode)

        self.seg_heads = args.seg.heads
        self.cls_heads = args.cls.heads

        self.recipe = args.trainer.recipe
        self._print_stats()

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        idx = self.df.index[index]
        input = self._get_input(idx)
        targets = self._get_target(idx)
        real_label = self.df.real.loc[idx]

        if self.transforms is not None:
            try:
                input, targets = self.transforms.apply_transformations(input, targets)
            except Exception as e:
                print(
                    f"=== \n error during transforms in path: {self.df.loc[idx, 'fpath']} \n {e} \n ==="
                )

        return {"idx": idx, "input": input, "target": targets, "real_label": real_label }

    def _get_input(self, idx):
        self.img_path =  self.img_folder_path[0] 
        for img_path in self.img_folder_path : 
            if os.path.exists(os.path.join(img_path, idx + ".png")) : 
                self.img_path = img_path
                break 
        path = os.path.join(self.img_path, idx + ".png")
        return read_image_from_path(path)

    def _get_scan_target(self, idx):
        dict = {}
        for cls in self.cls_heads:
            dict[cls] = self.df.loc[idx, cls]
        return dict

    def _get_age_target(self, idx):
        age_target = self.df.loc[idx, "age"]
        return age_target

    def _get_pixel_target(self, idx):
        """checks if annotation masks are present. If missing checks if seg_tag gt = 1,
        if tag is present then returns a masks with all vals -100  else returns masks with all vals 0
        """
        temp = {}
        ## get correct ann path
        # self.ann_path =  self.annotation_path[0] 
        # for ann_path in self.annotation_path : 
        #     if os.path.exists(os.path.join(self.ann_path, idx + ".png")) : 
        #         self.ann_path = ann_path
        #         break 
            

        for each_class in self.seg_heads:
            try:
                path_exists = False
                for ann_path in self.annotation_path:
                    annot_path = os.path.join(
                        ann_path, each_class, idx + ".png"
                    )
                    if os.path.exists(annot_path):
                        # read the output from the mask path and returns it
                        annot_path = annot_path
                        path_exists = True
                        break
               
                    # read the output from the mask path and returns it
                if path_exists:
                    temp[each_class] = read_image_from_path(annot_path)
                if not path_exists:
                    if ("cls" in self.recipe) and (each_class in self.cls_heads):
                        cls_target = self._get_scan_target(idx)

                        if cls_target[each_class] == 1:
                            # the cls is 1 but no mask so we give -100 to not propogate the loss for segmentation.
                            temp[each_class] = np.full(
                                (self.im_size, self.im_size), -100
                            )
                        else:
                            # this is performed if the target is 0 (not the abnormality , hence it gives a blank mask)
                            temp[each_class] = np.full((self.im_size, self.im_size), 0)
                    else:
                        """
                        this should get executed if only seg is performed on a class with no cls
                        if we want only segmentation and if the image path is not present and we have no
                        information if its normal or nota since we dont have the cls information , hence
                        we do not pass it for back propogation.
                        """
                        temp[each_class] = np.full((self.im_size, self.im_size), -100)
            except Exception as e:
                print(f"error in _get_pixel_target {e}")
                temp[each_class] = np.full((self.im_size, self.im_size), 0)

            """thresholding to convert mask into 0 or 1."""
            if temp[each_class].min() >= 0:
                temp[each_class] = (
                    (temp[each_class] > self.mask_threshold) * 1
                ).astype(dtype=np.int64)
            else:
                temp[each_class] = (temp[each_class]).astype(dtype=np.int64)

        return temp

    def _get_target(self, idx):
        self.targets = {}
        if "cls" in self.recipe:
            self.targets["classification_target"] = self._get_scan_target(idx)

        if "seg" in self.recipe:
            self.targets["segmentation_target"] = self._get_pixel_target(idx)

        if "age" in self.recipe:
            self.targets["age_target"] = self._get_age_target(idx)

        return self.targets

    def _print_stats(self):
        df = self.df
        print("total number of dataset values : ", len(df))

        print("number of cls positives")
        for cls in self.cls_heads:
            print(cls, " : ", df[cls].tolist().count(1))

        # print("total number of valid age values : ", len(df[df["is_age"] == 1]))

        print("number of segmentation masks")
        seg_classes = self.seg_heads
        for seg in seg_classes:
            if seg in df.columns.tolist():
                print(f"{seg} present in columns")
                dfpos = df[df[seg] == 1]
                posidxs = dfpos.index.tolist()

                exists = 0
                for path in self.annotation_path:
                    exists += sum(
                        [
                            os.path.exists(
                                os.path.join(path, seg, f"{idx}.png")
                            )
                            * 1
                            for idx in posidxs
                        ]
                    )
                print(seg, " : ", exists)

            else:
                seg_count = 0
                for path in self.annotation_path:
                    seg_count += len(glob.glob(os.path.join(path, seg, "*")))
                print(seg, " : ", seg_count)