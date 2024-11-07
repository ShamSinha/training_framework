import hashlib
import cv2
import os
import pandas as pd
import numpy as np
from torch.nn import Softmax as softmax


def read_image_from_path(path):
    try:
        im = cv2.imread(path, 0)
        ## resize the image to 960x960
        im = cv2.resize(im, (960, 960))
        if (im is None) or im.size == 0:
            raise RuntimeError("image array is either none or image.size is 0")
        im_resized = cv2.resize(im, (960, 960))
        return im_resized
    
    except Exception as e:
        print(f"error reading image from {path}, {e}")
        return np.zeros((100, 100))


def Filter_string_list_matching_substring_list(string, substr):
    """
    Filters any string that matches any substring

    eg:
    string = ['city1', 'class5', 'room2', 'city2']
    substr = ['class', 'city']
    print(Filter(string, substr))

    ['city1', 'class5', 'city2']
    """
    return [str for str in string if any(sub in str for sub in substr)]


def get_preds_from_logits(model_out, args):
    preds = {"classification_out": {}}

    for tag in args.classification.cls_heads:
        preds["classification_out"][tag] = apply_softmax(
            model_out["classification_out"][tag], "cls"
        )

    return preds


def apply_softmax(model_out, mode: str = "cls"):
    if mode == "seg":
        return softmax(dim=1)(model_out)[:, 1, ...]
    else:
        return softmax(dim=1)(model_out)


class Get_split:
    def __init__(
        self, module_int: int, split1_type: str, split2_type: str, index=-1
    ) -> None:
        """_summary_
            class function to split using the sha1 hash of the string s \n
            For 1 it would give split1_type with a prob of 1 \n
            for 2 it would give split1_type with a prob of 0.56 \n
            for 3 it would give split1_type with a prob of 0.375\n
            for 4 it would give split1_type with a prob of 0.3125 \n
            for 5 it would give split1_type with a prob of 0.25 \n
            for 6-8 it would give split1_type with a prob of 0.1875 \n
            for 9-16 it would give split1_type with a prob of 0.125 \n
            for >16 it would give split1_type with a prob of 0.0625 \n
        Args:
            module_int (int): the integer using which the module will be taken
            split1_type (str): the string returned if its divisible
            split2_type (str): the string returned if its not divisible
        """

        self.module_int = module_int
        if self.module_int == 0:
            raise Exception(
                "Sorry, zero can't be given as '0%0' leads to ZeroDivisionError:"
            )
        self.split1_type = split1_type
        self.split2_type = split2_type
        self.index = index

    def split_fn(self, s, index=-1):
        """
        Return the sha1 hash of the string s modulo 5.
        This is used to run the train test split
        To get the train val we use the (Secure Hash Algorithm 1) and we get the hex form of it
        ex : 'ca.phase1.unit1.00036234143b3d11be09bd19d5aeb187b7e66c9fdf2a6b96a83a4d22'

        gets converted to ex form to ae57b591f5ababc62ef8c8b0a9ac2b58a65e19f4

        then we take the last term and do modulo 9

        if the last digit is 0 then val else train , in this case since 0 can appear 2 times from (0 to 16)module9 the
        prob is 2/16 = 12.5% val

        """
        sha1mod8 = (
            int(hashlib.sha1(s.encode()).hexdigest()[self.index], 16) % self.module_int
        )
        if sha1mod8 == 0:
            return self.split1_type
        return self.split2_type


def get_images_in_dataset(img_folder_path):
    """
    Returns the images present in the image folder

    os.listdir is faster than glob and scandir for this particular case

    Example::

            import os, time, glob

            t = time.time()
            im_paths = glob.glob(f"{args.files.img_folder_path}/*")
            t = time.time() - t
            print ("glob.glob: %.4fs, %d files found" % (t, len(im_paths)))

            t = time.time()
            im_paths = os.listdir(args.files.img_folder_path)
            t = time.time() - t
            print ("os.listdir: %.4fs, %d files found" % (t, len(im_paths)))

            t = time.time()
            im_paths = None
            obj = os.scandir(args.files.img_folder_path)
            for entry in obj:
                im_paths = append(entry.name, im_paths)

            t = time.time() - t
            print ("os.scandir: %.4fs, %d files found" % (t, len(im_paths)))

    Output::
    >>> If we run this we get
    >>> glob.glob: 12.7882s, 1380328 files found
    >>> os.listdir: 10.8287s, 1380328 files found
    >>> os.scandir: 11.3973s, 1380328 files found
    """

    print("scanning image paths")
    im_paths = [os.path.join(img_path, file) for img_path in img_folder_path for file in os.listdir(img_path)]

    # this is done so that it matches with the ids of the csv dataframe files
    im_paths = {os.path.basename(x).replace(".png", ""): x for x in im_paths}
    print(f"no of images found: {len(im_paths)}  \n ")

    return pd.DataFrame.from_dict(im_paths, orient="index", columns=["filename"])


def read_csv(csv_path, class_heads, sources , use_3d_2d_fake_data_sampling ):
    """
    Get the respective columns and adds the filenames as index and adds nota columns
    which is the value that gives the oppsoite value of the target value

    >>> df["nota"] = df.sum(axis=1).apply(lambda x: 1 if x == 0 else 0)
    Example::
    >>> df
            opacity pleural effusion nota
        K0 	0       1                   0
        K1 	0       0                   1
        K2 	1       0                   0
        K3 	1       0                   0
        K4 	0       0                   1
    """
    dfcols = pd.read_csv(csv_path, nrows=1).columns.tolist()
    filtered_dfcols = [dfcols[0]]
    extra_tags = []
    if use_3d_2d_fake_data_sampling : 
        extra_tags = extra_tags + ["data_3d"]
    filtered_dfcols += set(list(set(class_heads + ["normal"] + ["real"] + extra_tags )))  ## make is config based 
    print("filtered_dfcols: " ,filtered_dfcols)

    df = pd.read_csv(csv_path, index_col=0, usecols=filtered_dfcols)
    # df["is_age"] = df["age"].apply(lambda x: 1 if x != -100 else 0)
    # nota is needed to create the sampling weights and while loading we are removing the nota
    # nota is also used as a tag while training
    # since pediatric is not a condition and its a property of the chest xray
    df["nota"] = (
        df[list(set(class_heads + ["normal"]))]
        .sum(axis=1)
        .apply(lambda x: 1 if x == 0 or x % -100 == 0 else 0)
    )

    """filtering sources"""
    # df["source"] = df.index.str.split(".", n=1).str[0]
    # if len(sources) > 0:
    #     df = df[df["source"].isin(sources)]

    # print(f"--- \n list of all sources used: {df.source.unique()}")

    return df


def train_val_split(df: pd.DataFrame, val_train_split_num=9):
    final_df = df.copy(deep=True)
    # to prevent the SettingWithCopyWarning:
    # (https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy)
    print("running the train val split")
    train_val_split = Get_split(val_train_split_num, "val", "train", -2)
    final_df["type"] = final_df.index.to_series().apply(
        lambda x: train_val_split.split_fn(x)
    )
    print(
        f'The number of training data points are : {len(final_df[final_df["type"] == "train"])}'
    )
    print(
        f'percentage: {(len(final_df[final_df["type"] == "train"])/len(final_df))*100} % \n '
    )
    print(
        f'The number of validation data points are : {len(final_df[final_df["type"] == "val"])}'
    )
    print(
        f'percentage: {(len(final_df[final_df["type"] == "val"])/len(final_df))*100} %  \n '
    )
    return final_df