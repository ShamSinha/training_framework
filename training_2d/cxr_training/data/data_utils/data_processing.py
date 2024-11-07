import pandas as pd
from cxr_training.data.data_utils.utils import (
    train_val_split,
    read_csv,
    get_images_in_dataset,
)
from cxr_training.data.data_utils.sampling_old import (
    Sampling_weight,
)


class Data_metadata:
    def __init__(self, args) -> None:
        """
        For testing purposes u can run

        Example::

        To get all the images files
        >>> from cxr_training.data.data_utils.data_processing import get_processed_metadata
        >>> valid_image_df = get_processed_metadata(args).get_all_images()

        To get the ground truth csv files with sampling u can run
        >>> from cxr_training.data.data_utils.data_processing import get_processed_metadata
        >>> gt_df = get_processed_metadata(args).get_GT_from_csv()

        To get only the csv files with the respective columns from cls_head with nota column
        >>> from cxr_training.data.data_utils.utils import  read_csv
        >>> df = read_csv(csv_path,class_heads)
        """
        self.args = args
        self.csv_path = args.files.ground_truth_csv
        self.cls_heads = args.cls.heads
        self.seg_heads = args.seg.heads
        self.sources = args.params.sources
        self.use_3d_2d_fake_data_sampling = args.params.use_3d_2d_fake_data_sampling
        self.img_folder_path = args.files.img_folder_path

    def _get_all_images(self):
        return get_images_in_dataset(self.img_folder_path)

    def _get_GT_from_csv(self):
        self.df = read_csv(self.csv_path, self.cls_heads, self.sources , self.use_3d_2d_fake_data_sampling )
        """
        for each_class in self.cls_heads:
            print(
                f"invalid data in {each_class} is {len(self.df.loc[(self.df[each_class]!=0)&(self.df[each_class]!=1)])}"
            )
            self.df = self.df.loc[
                (self.df[each_class] == 0) | (self.df[each_class] == 1)
            ]

        print(
            f"the total number of images present in the ground truth csv is : {len(self.df)} \n "
        )
        """
        return self.df


class ImageLabel(Data_metadata):
    def __init__(self, args):
        super().__init__(args)
        self.sampler = args.params.sampler

    def get_data(self):
        """
        filtering ids for which valid .png is present
        Selects the index of the two dataframes for joining , similar to
        pd.merge(left, right, how = 'inner', on = ['indexname1', 'indexname2'])
        If a image is present in gt_df but not present in valid_image_df then it wont show in the final_df

        Example::
        >>> gt_df
                opacity
            K0 	0
            K1 	0
            K2 	1
            K3 	1
            K4 	0

        >>> valid_image_df
                    filename
            K0 	K0
            K1 	K1
            K2 	K2
            K3 	K3
            K5  K5

        >>> gt_df.merge(valid_image_df, "inner", left_index=True, right_index=True)
                    opacity filename
                K0 	0 	K0
                K1 	0 	K1
                K2 	1 	K2
                K3 	1 	K3

        """
        gt_df = self._get_GT_from_csv()
        valid_image_df = self._get_all_images()
        final_df = gt_df.merge(
            valid_image_df, "inner", left_index=True, right_index=True
        )
        final_df = train_val_split(final_df)
        sampling_function = Sampling_weight(final_df, self.args)
        final_df = sampling_function.all_sampling_weights()

        return final_df


class EntireImages(Data_metadata):
    def __init__(self, args):
        super().__init__(args)

    def get_data(self):
        """
        gets the database with the images and if it dosent have a ground truth it is present as null

        Using the above example

        >>> valid_image_df.merge(gt_df, "left", left_index=True, right_index=True)
                filename 	opacity
            K0 	    K0 	       0
            K1 	    K1 	       0
            K2 	    K2 	       1
            K3 	    K3 	       1
            K5 	    K5 	       NaN

        """
        gt_df = self._get_GT_from_csv()
        valid_image_df = self._get_all_images()
        final_df = valid_image_df.merge(
            gt_df, "left", left_index=True, right_index=True
        )
        final_df = train_val_split(final_df)

        return final_df
