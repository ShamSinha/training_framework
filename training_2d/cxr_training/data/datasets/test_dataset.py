from cxr_training.data.data_utils.utils import read_image_from_path
from cxr_training.data.transforms.transform_controller import (
    Test_Augment,
    TestTimeAugment,
)
import pandas as pd
from torch.utils.data import Dataset
import os
import glob


class Test_dataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.img_folder_path = self.args.files.testing_images
        self.classes = self.args.cls.heads
        self.csv_path = self.args.files.testing_csv
        self.im_size = self.args.params.im_size
        self.use_subset = self.args.params.inference_on_subset
        self.inference_type = self.args.params.inference_type

        if "tta" in self.inference_type:
            self.transforms = TestTimeAugment(self.args)

        else:
            self.transforms = Test_Augment(self.args)

        self.only_images = self.args.params.inference_on_images

        if self.only_images:
            self.read_only_images()
        else:
            self.valid_images()
            self.read_csv()
            self._print_stats()

    def __len__(self):
        return len(self.df.index)

    def read_only_images(self):
        self.df = pd.DataFrame()
        images  = [file for img_path in self.img_folder_path for file in os.listdir(img_path)]
        # images = [os.listdir(img_path) for img_path in self.img_folder_path]
        # images = os.listdir(self.img_folder_path)
        self.df["file_name"] = images
        self.df.index = [x[:-4] for x in images]

    def valid_images(self):
        self.im_paths = []
        print("globbing image paths")
        for folder in self.img_folder_path:
            self.im_paths.extend(glob.glob(f"{folder}/*"))

        # this is done so that we check and select all the matching images from the pandas dataframe file
        self.im_paths = {
            os.path.basename(x).replace(".png", ""): x for x in self.im_paths
        }
        print(f"no of images found: {len(self.im_paths)}")
        self.valid_image_df = pd.DataFrame.from_dict(
            self.im_paths, orient="index", columns=["fpath"]
        )

    def read_csv(self):
        csv_path = self.csv_path
        filtered_dfcols = ["filename"] + self.classes
        dtypes = {x: int for x in self.classes}
        dtypes["filename"] = str
        df = pd.read_csv(csv_path, index_col=0, usecols=filtered_dfcols, dtype=dtypes)

        df["nota"] = df.sum(axis=1).apply(lambda x: 1 if x == 0 else 0)

        df = df[~df.index.duplicated(keep=False)]
        if self.use_subset:
            print("using only a subset of the test dataset")
            sample_abnormal = len(df[(df["normal"] == 0) & (df["nota"] == 0)]) // 10
            sample_normal = len(df[df["normal"] == 1]) // 10
            sample_nota = len(df[df["nota"] == 1]) // 10

            sample_abnormal = max(sample_abnormal, 5000)
            sample_normal = max(sample_normal, 5000)
            sample_nota = max(sample_nota, 5000)

            df_abnormal = df[(df["normal"] == 0) & (df["nota"] == 0)].sample(
                n=sample_abnormal, random_state=1
            )

            print(
                f"\n \n \nRunning cls on subset. abnormals {len(df_abnormal)}, "
                f"normals: {sample_normal} notas: {sample_nota}"
            )
            df_normal = df[df["normal"] == 1].sample(n=sample_normal, random_state=1)
            df_nota = df[df["nota"] == 1].sample(n=sample_nota, random_state=1)
            df = pd.concat([df_normal, df_nota, df_abnormal], ignore_index=False)

        # merget the two df to get the common images that have labels in the csv file
        self.df = df.merge(
            self.valid_image_df, "inner", left_index=True, right_index=True
        )
        total_parts = getattr(self.args.params, "total_parts", -1)
        part = getattr(self.args.params, "part_num", -1)
        if total_parts != -1:
            print(f"total_parts: {total_parts}, part: {part}")
            ## take fraction of the data
            total_len = len(self.df)
            start = (total_len // total_parts) * (part-1)
            if part == total_parts:
                end = total_len
            else:
                end = (total_len // total_parts) * (part)
            self.df = self.df.iloc[start:end].copy()
            print(f"part: {part}, len: {len(self.df)}")

    def __getitem__(self, index):
        idx = self.df.index[index]

        input = self._get_input(idx)

        if not self.only_images:
            targets = self._get_target(idx)

        if self.transforms is not None:
            try:
                input = self.transforms(input)
            except Exception as e:
                print(
                    f"=== \n error during transforms in path: {self.df.loc[idx, 'fpath']} \n {e} \n ==="
                )
        if self.only_images:
            return {"idx": idx, "input": input}

        return {"idx": idx, "input": input, "target": targets}

    def _get_input(self, idx):
        # path = os.path.join(self.img_folder_path, idx + ".png")
        self.img_path =  self.img_folder_path[0] 
        for img_path in self.img_folder_path : 
            if os.path.exists(os.path.join(img_path, idx + ".png")) : 
                self.img_path = img_path
                break 
        path = os.path.join(self.img_path, idx + ".png")
        return read_image_from_path(path)

    def _get_scan_target(self, idx):
        return dict(self.df.loc[idx, self.classes])

    def _get_target(self, idx):
        self.targets = {}
        self.targets["classification_target"] = self._get_scan_target(idx)
        return self.targets

    def _print_stats(self):
        df = self.df
        cls_classes = self.classes

        print("total number of dataframe values : ", len(df))

        print("number of cls positives")
        for cls in cls_classes:
            print(cls, " : ", df[cls].tolist().count(1))

        print("number of cls negatives")
        for cls in cls_classes:
            print(cls, " : ", len(df) - df[cls].tolist().count(1))

    def get_test_df(self):
        return self.df
