from typing import List
from tqdm import tqdm
from cxr_training.inference.inference_nnmodule_controller import (
    load_model,
    get_dataloader,
    set_cuda_gpus,
)
from send2trash import send2trash
from omegaconf import OmegaConf
import os
import torch
import pandas as pd
import torch.nn.functional as fn
from numpy import uint8
from cv2 import imwrite
import json
import glob


class run_inference_for_model_list:
    def __init__(
        self,
        model_list: List,
        args,
    ):
        self.model_list = model_list
        self.args = args
        total_parts = getattr(self.args.params, "total_parts", -1)
        if total_parts != -1:
            part = self.args.params.part_num
            self.csv_suffix = f"_part_{part}_of_{total_parts}"
        else:
            self.csv_suffix = ""
        self.gpus = (
            list(range(torch.cuda.device_count()))
            if args.trainer.gpus == [-1]
            else args.trainer.gpus
        )
        self.checkpoint_dir = self.args.trainer.checkpoint_dir
        self.seg_heads = self.args.seg.heads
        self.description = self.args.trainer.description
        self.model_file = self.args.trainer.model_file
        self.cls_heads = self.args.cls.heads
        self.inference_type = self.args.params.inference_type
        self.only_images = self.args.params.inference_on_images
        self.test_annotation_path = self.args.files.testing_annotation_path
        self.test_data_type = self.args.files.test_data_type
        self.results_folder = None
        self.model_name = None
        self.current_model_path = None
        self.dataloader = None
        self.auc_csv_fpath = None

    def run_inference(self) -> None:
        for model_path in tqdm(self.model_list):
            print("Model : ",model_path )
            self.current_model_path = model_path
            set_cuda_gpus(self.gpus)
            self.load_args_and_model()
            self.get_results_path()
            self.configure_dataloader()
            
            if self.current_model_path.endswith(".ts"):
                self.run_classification_segmentation_ts()
            elif "cls_seg_comb" in self.inference_type:
                self.run_classification_segmentation()
            else:
                if "cls" in self.inference_type:
                    self.run_classification()

                if "seg" in self.inference_type:
                    self.run_segmentation()

    def load_args_and_model(self) -> None:
        if self.current_model_path.endswith(".ts"):
            self.model = torch.jit.load(self.current_model_path)
            self.model.eval()
        else:
            self.model, args = load_model(self.current_model_path)

    def configure_dataloader(self) -> None:
        self.dataloader = get_dataloader(self.args)

    def get_results_path(self, suffix="") -> None:
        self.model_name = os.path.basename(self.current_model_path).replace(
            "." + f'{self.current_model_path.split(".")[-1]}', ""
        )

        self.results_folder = f"{self.checkpoint_dir}/results/{self.description}/{self.model_file}/{suffix}/{self.model_name}"

        os.makedirs(self.results_folder, exist_ok=True)

        with open(f"{self.results_folder}/args.json", "w") as outfile:
            json.dump(OmegaConf.to_container(self.args), outfile)

        self.auc_csv_fpath = os.path.join(
            self.results_folder, self.model_name + f"{self.csv_suffix}.csv"
        )

    def run_classification(self) -> None:
        self.get_results_path(suffix= self.test_data_type)
        
        if os.path.exists(self.auc_csv_fpath) and os.path.isfile(self.auc_csv_fpath):
            raise FileExistsError(f"File '{self.auc_csv_fpath}' exists. Stopping execution.")


        data = []

        for _idx, batch in enumerate(tqdm(self.dataloader)):
            indices, input, targets = map(batch.get, ("idx", "input", "target"))

            batch_data = pd.DataFrame({"filename": indices})
            with torch.no_grad():
                preds = self.model(input.to(device="cuda"))

            age_inference = False
            if len(preds["age_out"]) > 0:
                age_inference = True

            for each_class in self.cls_heads:
                batch_data[each_class] = pd.Series(
                    fn.softmax(preds["classification_out"][each_class], dim=1)[:, 1]
                    .detach()
                    .cpu()
                    .numpy()
                )

                if not self.only_images:
                    if each_class in targets["classification_target"]:
                        batch_data[each_class + "_target"] = pd.Series(
                            targets["classification_target"][each_class]
                        )
            if age_inference:
                batch_data["age"] = pd.Series(
                    preds["age_out"].view(-1).detach().cpu().numpy()
                )

            data.append(pd.DataFrame(batch_data))

        df = pd.concat(data)
        df.to_csv(self.auc_csv_fpath, index=False)

    def run_segmentation(self):
        self.get_results_path(suffix= self.test_data_type)
        tags = [
            os.path.basename(t)
            for path in self.test_annotation_path
            for t in glob.glob(f"{path}/*")
        ]
        valid_tags = set(tags).intersection(set(self.seg_heads))

        print("valid_tags:", valid_tags)

        gt_dir = {}
        for t in valid_tags:
            gt_dir[t] = []  # Initialize an empty list for each tag
            for path in self.test_annotation_path:
                full_path = f"{path}/{t}/"
                if os.path.exists(full_path):  # Check if the path exists
                    gt_dir[t].extend([x[:-4] for x in os.listdir(full_path)])  
                    
        for key in gt_dir:
            print(f"{key}: {len(gt_dir[key])} items")

        data = []
        for _idx, batch in enumerate(tqdm(self.dataloader)):
            indices, input, targets = map(batch.get, ("idx", "input", "target"))
            batch_data = pd.DataFrame({"filename": indices})
            with torch.no_grad():
                preds = self.model(input.to(device="cuda"))
            for each_class in self.seg_heads:
                res_dir = os.path.join(self.results_folder, each_class)
                os.makedirs(res_dir, exist_ok=True)
                seg = (
                    fn.softmax(preds["segmentation_out"][each_class], dim=1)[:, 1, ...]
                    .cpu()
                    .numpy()
                )
                max_pixel_scores = []
                average_pixel_scores = []
                count_pixel_scores = []
                for i in range(seg.shape[0]):
                    max_pixel_score = seg[i].max()  # Get the max pixel value
                    average_pixel_score = seg[i].mean()
                    count_pixel_score = seg[i].sum()
                    max_pixel_scores.append(max_pixel_score)
                    average_pixel_scores.append(average_pixel_score)
                    count_pixel_scores.append(count_pixel_score)
                    img = (seg[i] * 255).astype(uint8)
                    if indices[i] in gt_dir[each_class]:
                        filename = os.path.join(res_dir, indices[i] + ".png")
                        imwrite(filename, img)

                if not self.only_images:
                    if each_class in targets["classification_target"]:
                        batch_data[each_class + "_target"] = pd.Series(
                            targets["classification_target"][each_class]
                        )

                batch_data[each_class + "_max_pixel_score"] = max_pixel_scores
                batch_data[each_class + "_average_pixel_score"] = average_pixel_scores
                batch_data[each_class + "_count_pixel_score"] = count_pixel_scores
            data.append(pd.DataFrame(batch_data))

        df = pd.concat(data)
        df.to_csv(self.auc_csv_fpath.replace(".csv", "_pixel_scores.csv"), index=False)

    def run_classification_segmentation(self):
        self.get_results_path(suffix= self.test_data_type)
        
        tags = [
            os.path.basename(t)
            for path in self.test_annotation_path
            for t in glob.glob(f"{path}/*")
        ]
        valid_tags = set(tags).intersection(set(self.seg_heads))

        print("valid_tags:", valid_tags)
        
        gt_dir = {}
        for t in valid_tags:
            gt_dir[t] = []  # Initialize an empty list for each tag
            for path in self.test_annotation_path:
                full_path = f"{path}/{t}/"
                if os.path.exists(full_path):  # Check if the path exists
                    gt_dir[t].extend([x[:-4] for x in os.listdir(full_path)])  
                    
        for key in gt_dir:
            print(f"{key}: {len(gt_dir[key])} items")

        if os.path.exists(self.auc_csv_fpath) and os.path.isfile(self.auc_csv_fpath):
            raise FileExistsError(f"File '{self.auc_csv_fpath}' exists. Stopping execution.")
        
        # this is to make sure it always remain open
        f = open(self.auc_csv_fpath, "a")        

        for idx, batch in enumerate(tqdm(self.dataloader)):
            indices, input, targets = map(batch.get, ("idx", "input", "target"))
            df = pd.DataFrame({"filename": indices})

            with torch.no_grad():
                preds = self.model(input.to(device="cuda"))

            age_inference = False
            if len(preds["age_out"]) > 0:
                age_inference = True

            for each_class in self.cls_heads:
                df[each_class] = pd.Series(
                    fn.softmax(preds["classification_out"][each_class], dim=1)[:, 1]
                    .detach()
                    .cpu()
                    .numpy()
                )

                if not self.only_images:
                    if each_class in targets["classification_target"]:
                        df[each_class + "_target"] = pd.Series(
                            targets["classification_target"][each_class]
                        )

            if age_inference:
                df["age"] = pd.Series(preds["age_out"].view(-1).detach().cpu().numpy())
                
            for each_class in self.seg_heads:
                res_dir = os.path.join(self.results_folder, each_class)
                os.makedirs(res_dir, exist_ok=True)
                seg = (
                    fn.softmax(preds["segmentation_out"][each_class], dim=1)[:, 1, ...]
                    .cpu()
                    .numpy()
                )
                max_pixel_scores = []
                average_pixel_scores = []
                count_pixel_scores = []
                for i in range(seg.shape[0]):
                    max_pixel_score = seg[i].max()  # Get the max pixel value
                    average_pixel_score = seg[i].mean()
                    count_pixel_score = seg[i].sum()
                    max_pixel_scores.append(max_pixel_score)
                    average_pixel_scores.append(average_pixel_score)
                    count_pixel_scores.append(count_pixel_score)
                    img = (seg[i] * 255).astype(uint8)
                    # if indices[i] in gt_dir[each_class]:
                    filename = os.path.join(res_dir, indices[i] + ".png")
                    imwrite(filename, img)
                        
                # print(df.shape, len(max_pixel_scores), len(average_pixel_scores), len(count_pixel_scores), seg.shape)
                df[each_class + "_max_pixel_score"] = max_pixel_scores
                df[each_class + "_average_pixel_score"] = average_pixel_scores
                df[each_class + "_count_pixel_score"] = count_pixel_scores
            # mode='a'appends to the end of the csv files,warning:if the csv files already exist delete them before hand
            df.to_csv(f, header=(idx == 0))
        f.close()
    
    def run_classification_segmentation_ts(self):
        self.get_results_path(suffix= self.test_data_type)
        
        tags = [
            os.path.basename(t)
            for path in self.test_annotation_path
            for t in glob.glob(f"{path}/*")
        ]
        valid_tags = set(tags).intersection(set(self.seg_heads))

        print("valid_tags:", valid_tags)
        
        gt_dir = {}
        for t in valid_tags:
            gt_dir[t] = []  # Initialize an empty list for each tag
            for path in self.test_annotation_path:
                full_path = f"{path}/{t}/"
                if os.path.exists(full_path):  # Check if the path exists
                    gt_dir[t].extend([x[:-4] for x in os.listdir(full_path)])  
                    
        for key in gt_dir:
            print(f"{key}: {len(gt_dir[key])} items")

        if os.path.exists(self.auc_csv_fpath) and os.path.isfile(self.auc_csv_fpath):
            raise FileExistsError(f"File '{self.auc_csv_fpath}' exists. Stopping execution.")
        
        # this is to make sure it always remain open
        f = open(self.auc_csv_fpath, "a")        

        for idx, batch in enumerate(tqdm(self.dataloader)):
            indices, input, targets = map(batch.get, ("idx", "input", "target"))
            df = pd.DataFrame({"filename": indices})

            with torch.no_grad():
                preds_ts = self.model(input.to(device="cuda"))
            preds = {}
            preds["classification_out"] = preds_ts[0]
            preds["segmentation_out"] = preds_ts[1]
            age_inference = False
            # if len(preds["age_out"]) > 0:
            #     age_inference = True

            for each_class in self.cls_heads:
                df[each_class] = pd.Series(
                    preds["classification_out"][each_class]
                    .detach()
                    .cpu()
                    .numpy()
                )

                if not self.only_images:
                    if each_class in targets["classification_target"]:
                        df[each_class + "_target"] = pd.Series(
                            targets["classification_target"][each_class]
                        )

            if age_inference:
                df["age"] = pd.Series(preds["age_out"].view(-1).detach().cpu().numpy())
                
            max_pixel_scores = []
            average_pixel_scores = []
            count_pixel_scores = []
            for each_class in self.seg_heads:
                res_dir = os.path.join(self.results_folder, each_class)
                os.makedirs(res_dir, exist_ok=True)
                seg = (
                    preds["segmentation_out"][each_class]
                    .cpu()
                    .numpy()
                )

                for i in range(seg.shape[0]):
                    max_pixel_score = seg[i].max()  # Get the max pixel value
                    average_pixel_score = seg[i].mean()
                    count_pixel_score = seg[i].sum()
                    max_pixel_scores.append(max_pixel_score)
                    average_pixel_scores.append(average_pixel_score)
                    count_pixel_scores.append(count_pixel_score)
                    img = (seg[i] * 255).astype(uint8)
                    # if indices[i] in gt_dir[each_class]:
                    filename = os.path.join(res_dir, indices[i] + ".png")
                    imwrite(filename, img)
                        
                df[each_class + "_max_pixel_score"] = max_pixel_scores
                df[each_class + "_average_pixel_score"] = average_pixel_scores
                df[each_class + "_count_pixel_score"] = count_pixel_scores
            # mode='a'appends to the end of the csv files,warning:if the csv files already exist delete them before hand
            df.to_csv(f, header=(idx == 0))
        f.close()