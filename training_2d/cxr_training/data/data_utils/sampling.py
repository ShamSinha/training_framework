import pandas as pd
import glob
import os
import math
import omegaconf

class Sampling_weight:
    def __init__(self, df, args) -> None:
        self.df = df
        self.cls_classes = args.cls.sampling_tags
        self.seg_classes = args.seg.sampling_tags
        self.equal_source_sampling = args.params.equal_source_sampling
        self.user_class_wts = args.cls.user_class_wts
        self.user_annotation_wts = args.seg.user_class_wts
        self.annotation_path = args.files.annotation_path
        self.recipe = args.trainer.recipe
        self.use_real_fake_sampling = args.params.use_real_fake_sampling
        self.use_3d_2d_fake_data_sampling = args.params.use_3d_2d_fake_data_sampling
        
        if self.use_real_fake_sampling:
            self.real_data_wts = args.params.real_data_wts
            
        if self.use_3d_2d_fake_data_sampling: 
            self.data_weights_3d = args.params.data_weights_3d

        if self.recipe == "cls_seg":
            if any(x in self.seg_classes for x in ["normal", "nota"]) and any(
                x in self.cls_classes for x in ["normal", "nota"]
            ):
                print(
                    "normal and nota present as sampling in both cls and seg but its enough in only one of them"
                )

        self.df["final_sample_wts"] = 0

    def get_cls_sampling_wts(self):
        """_
        The sampling weights that would be used by the weighted random sampler present in the dataloader
        """
        print("Computing Classification weights")
        class_sampling_wts_dict = {}
        for each_class in self.cls_classes:
            class_sampling_wts_dict[each_class] = 1 / math.sqrt(
                self.df.loc[self.df[each_class] != -100, each_class].sum() + 0.000001
            )
            print(
                f"base class probability based on gts for {each_class} is {class_sampling_wts_dict[each_class]}  \n "
            )
        class_sampling_wts_dict = pd.Series(class_sampling_wts_dict)

        for each_class in self.cls_classes:
            print(
                f"{each_class} : {class_sampling_wts_dict[each_class]*self.user_class_wts[each_class]}"
            )
            self.df[each_class + "_class_sampling_wts"] = (
                (self.df[each_class].replace(to_replace=-100, value=0))
                * class_sampling_wts_dict[each_class]
                * self.user_class_wts[each_class]
            )
            self.df["final_sample_wts"] = (
                self.df["final_sample_wts"]
                + self.df[each_class + "_class_sampling_wts"]
            )

    def _annotation_ids(self):
        all_annot_ids = {}
        for each_class in self.seg_classes:
            if "normal" in each_class or "nota" in each_class:
                continue

            # all_annots = glob.glob(f"{self.annotation_path}/{each_class}/*")
            if isinstance(self.annotation_path,  omegaconf.listconfig.ListConfig):
                all_annots = []
                for path in self.annotation_path:
                    all_annots.extend(glob.glob(f"{path}/{each_class}/*"))
            else:
                all_annots = glob.glob(f"{self.annotation_path}/{each_class}/*")
                
            all_annot_ids[each_class] = {
                os.path.basename(x).replace(".png", "") for x in all_annots
            }
        return all_annot_ids

    def _calculate_common_class_weights(self):
        # Calculate and apply common class weights for 'normal' and 'nota', if applicable
        is_annotation = False
        if "seg" in self.recipe:
            is_annotation = True
        for class_name in ["normal", "nota"]:
            if class_name in self.cls_classes or class_name in self.seg_classes:
                self._calculate_and_apply_weights(class_name, is_annotation)

    def _calculate_and_apply_weights(self, class_name, is_annotation=False):
        # Generalized weight calculation and application method
        base_weight = 1 / (
            self.df.loc[self.df[class_name] != -100, class_name].sum() + 1e-6
        )
        user_weight = (
            self.user_class_wts if not is_annotation else self.user_annotation_wts
        )
        self.df[f"{class_name}_weight"] = (
            self.df[class_name].replace(-100, 0)
            * base_weight
            * user_weight.get(class_name, 1)
        )
        self.df["final_sample_wts"] += self.df[f"{class_name}_weight"]

    def get_seg_sampling_wts(self):
        print("Computing Segmentation weights")
        self.df["max_annotation_weights"] = 0

        seg_sampling_wts_dict = {}
        for each_class in [c for c in self.seg_classes if c not in ["normal", "nota"]]:
            total_files = 0
            for path in self.annotation_path:
                total_files += len(os.listdir(f"{path}/{each_class}/"))

            if total_files > 0:
                seg_sampling_wts_dict[each_class] = 1 / math.sqrt(total_files)
            else:
                seg_sampling_wts_dict[each_class] = 0  # Handle case where no files are found
                
            print(
                f"base seg probability based on gts for {each_class} is {seg_sampling_wts_dict[each_class]}  \n "
            )
        seg_sampling_wts_dict = pd.Series(seg_sampling_wts_dict)

        all_annot_ids = self._annotation_ids()
        for each_class in [c for c in self.seg_classes if c not in ["normal", "nota"]]:
            annot_available_factor = seg_sampling_wts_dict[each_class]
            user_annot_sampling_factor = self.user_annotation_wts.get(each_class, 1)
            annot_sampling_factor = annot_available_factor * user_annot_sampling_factor

            if not ("normal" in each_class or "nota" in each_class):
                self.df[each_class + "_annotations_sampling_wts"] = [
                    annot_sampling_factor if (x in all_annot_ids[each_class]) else 0
                    for x in self.df.index
                ]

            self.df["max_annotation_weights"] = (
                # If the current max weight is greater than or equal to the class-specific weight,
                # keep the current max weight.
                self.df["max_annotation_weights"]
                >= self.df[each_class + "_annotations_sampling_wts"]
            ) * self.df["max_annotation_weights"] + (
                # If the current max weight is less than the class-specific weight,
                # update it to the class-specific weight.
                self.df["max_annotation_weights"]
                < self.df[each_class + "_annotations_sampling_wts"]
            ) * self.df[
                each_class + "_annotations_sampling_wts"
            ]

        self.df["final_sample_wts"] = (
            self.df["final_sample_wts"] + self.df["max_annotation_weights"]
        )

    def get_source_sampling_wts(self):
        print("\033[91m" + "Warning: using equal source sampling." + "\033[0m")

        source_sampling_wts_dict = (
            self.df["source"].value_counts().apply(lambda x: (1 / x)).to_dict()
        )
        self.df["source_sampling_wts"] = self.df["source"].map(source_sampling_wts_dict)
        self.df["final_sample_wts"] = (
            self.df["final_sample_wts"] * self.df["source_sampling_wts"]
        )

        print(source_sampling_wts_dict)


    def get_real_fake_sampling_wts(self):

        total_nodule = self.df[(self.df['nodule'] == 1)].shape[0]
        print("value_counts of real : " ,self.df['real'].value_counts())
        
        real_nodule_wt_factor = total_nodule*self.real_data_wts/self.df[(self.df['real'] == 1) & (self.df['nodule'] == 1)].shape[0]
        fake_nodule_wt_factor = total_nodule*(1 - self.real_data_wts)/self.df[(self.df['real'] == 0) & (self.df['nodule'] == 1)].shape[0]
        
        print("real_nodule_wt_factor: " , real_nodule_wt_factor)
        print("fake_nodule_wt_factor: " , fake_nodule_wt_factor)
        
        self.df.loc[(self.df['real'] == 1) & (self.df['nodule'] == 1), 'final_sample_wts'] *= real_nodule_wt_factor
        self.df.loc[(self.df['real'] == 0) & (self.df['nodule'] == 1), 'final_sample_wts'] *= fake_nodule_wt_factor

    def get_3d_2d_fake_data_sampling_wts(self):
        total_fake_nodules = self.df[(self.df['nodule'] == 1) &  (self.df['real'] == 0)].shape[0]
        print("total_fake_nodules: " ,total_fake_nodules )
        
        fake_3d_nodule_wt_factor =  total_fake_nodules * self.data_weights_3d/self.df[(self.df['real'] == 0) & (self.df['nodule'] == 1) & (self.df['data_3d'] == 1) ].shape[0]
        fake_2d_nodule_wt_factor = total_fake_nodules * (1-self.data_weights_3d)/self.df[(self.df['real'] == 0) & (self.df['nodule'] == 1) & (self.df['data_3d'] == 0) ].shape[0]
        
        print("fake_3d_nodule_wt_factor: ", fake_3d_nodule_wt_factor)
        print("fake_2d_nodule_wt_factor: ", fake_2d_nodule_wt_factor)
        
        self.df.loc[(self.df['real'] == 0) & (self.df['nodule'] == 1) & (self.df['data_3d'] == 1), 'final_sample_wts'] *= fake_3d_nodule_wt_factor
        self.df.loc[(self.df['real'] == 0) & (self.df['nodule'] == 1) & (self.df['data_3d'] == 0), 'final_sample_wts'] *= fake_2d_nodule_wt_factor

    def all_sampling_weights(self):

        self._calculate_common_class_weights()
        if "cls" in self.recipe:
            self.get_cls_sampling_wts()
        if "seg" in self.recipe:
            self.get_seg_sampling_wts()

        
        if self.equal_source_sampling:
            self.get_source_sampling_wts()

        if self.use_real_fake_sampling:
            self.get_real_fake_sampling_wts()
            
        if self.use_3d_2d_fake_data_sampling:
            self.get_3d_2d_fake_data_sampling_wts()

        tr_df = self.df[self.df["type"] == "train"]
        print("Final Sampling Weights for each class/ sum of all weights")
        for each_class in self.cls_classes:
            print(each_class, tr_df[tr_df[each_class]==1]["final_sample_wts"].sum()/tr_df["final_sample_wts"].sum())
        return self.df
