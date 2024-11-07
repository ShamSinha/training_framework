from typing import List
import pandas as pd
import numpy as np
import numpy.ma as ma
from cxr_training.metrics.test_metric_utils import iou_dirs, get_cutoff_youdens_index
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
import os
from send2trash import send2trash
import glob


class Metrics:
    """
    Metrics class to calculate and generate various performance metrics
    for machine learning models, specifically designed for CXR models.

    Attributes:


    -----------
    model_list: List
        List of paths to models.
    args: object
        Arguments object containing various configuration parameters.
    ... (other attributes) ...
    """

    def __init__(self, model_list: List, args):
        """Initialize the Metrics object with models and arguments."""
        self.model_list = model_list
        self.args = args
        self._setup_paths_from_args()

    def _setup_paths_from_args(self):
        """Setup various paths using the arguments provided."""
        self.checkpoint_dir = self.args.trainer.checkpoint_dir
        self.description = self.args.trainer.description
        self.model_file = self.args.trainer.model_file
        self.test_annotation_dir = self.args.files.testing_annotation_path
        self.metric_type = self.args.params.metric_type
        self.beta = self.args.params.f1_beta_value
        self.test_data_type = self.args.files.test_data_type

    def get_auROC(self, filepath):
        self.model_tag_auc["AUROC"] = 0
        df = pd.read_csv(filepath)
        tags_with_targets = [
            t for t in df.columns.tolist() if f"{t}_target" in df.columns.tolist()
        ]
        print(f"tags used {tags_with_targets}")
        for t in tags_with_targets:
            df_tag = df[df[f"{t}_target"] != -100]
            try:
                self.model_tag_auc[f"{t}_AUROC"] = roc_auc_score(
                    df_tag[f"{t}_target"], df_tag[t]
                )
                fpr, tpr, thresholds = roc_curve(df_tag[f"{t}_target"], df_tag[t])
                self.model_tag_auc[
                    f"{t}_youden_index_cuttoff"
                ] = get_cutoff_youdens_index(fpr, tpr, thresholds)
                self.model_tag_auc["AUROC"] += self.model_tag_auc[f"{t}_AUROC"]
            except Exception as e:
                print(e)
                print("=" * 50, f"error while cal auroc for {t}", "=" * 50)
                self.model_tag_auc[t] = 0.1

    def get_auPRC(self, filepath):
        self.model_tag_auc["AUPRC"] = 0
        df = pd.read_csv(filepath)
        tags_with_targets = [
            t for t in df.columns.tolist() if f"{t}_target" in df.columns.tolist()
        ]

        for t in tags_with_targets:
            df_tag = df[df[f"{t}_target"] != -100]
            try:
                self.model_tag_auc[f"{t}_AUPRC"] = average_precision_score(
                    df_tag[f"{t}_target"], df_tag[t]
                )
                self.model_tag_auc["AUPRC"] += self.model_tag_auc[f"{t}_AUPRC"]
            except Exception as e:
                print(e)
                print("=" * 50, f"error while cal AUPRC for {t}", "=" * 50)
                self.model_tag_auc[t] = 0.1

    def get_F1score(self, filepath):
        self.model_tag_auc["F1_score"] = 0
        self.model_tag_auc["Fbeta_score"] = 0
        df = pd.read_csv(filepath)
        tags_with_targets = [
            t for t in df.columns.tolist() if f"{t}_target" in df.columns.tolist()
        ]

        for t in tags_with_targets:
            df_tag = df[df[f"{t}_target"] != -100]
            try:
                self.model_tag_auc[f"{t}_F1_score"] = fbeta_score(
                    df_tag[f"{t}_target"],
                    (df_tag[t] >= self.model_tag_auc[f"{t}_youden_index_cuttoff"]) * 1,
                    beta=1,
                )
                self.model_tag_auc[f"{t}_Fbeta_score"] = fbeta_score(
                    df_tag[f"{t}_target"],
                    (df_tag[t] >= self.model_tag_auc[f"{t}_youden_index_cuttoff"]) * 1,
                    beta=self.beta,
                )
                self.model_tag_auc["F1_score"] += self.model_tag_auc[f"{t}_F1_score"]
                self.model_tag_auc["Fbeta_score"] += self.model_tag_auc[
                    f"{t}_Fbeta_score"
                ]
                self.model_tag_auc["Fscore_beta"] = self.beta
            except Exception as e:
                print(e)
                print("=" * 50, f"error while cal F1score for {t}", "=" * 50)
                self.model_tag_auc[t] = 0.1

    def get_MCC(self, filepath):
        self.model_tag_auc["MCC"] = 0
        df = pd.read_csv(filepath)
        tags_with_targets = [
            t for t in df.columns.tolist() if f"{t}_target" in df.columns.tolist()
        ]

        for t in tags_with_targets:
            df_tag = df[df[f"{t}_target"] != -100]
            try:
                self.model_tag_auc[f"{t}_MCC"] = matthews_corrcoef(
                    df_tag[f"{t}_target"],
                    (df_tag[t] >= self.model_tag_auc[f"{t}_youden_index_cuttoff"]) * 1,
                )
                self.model_tag_auc["MCC"] += self.model_tag_auc[f"{t}_MCC"]
            except Exception as e:
                print(e)
                print("=" * 50, f"error while cal MCC for {t}", "=" * 50)
                self.model_tag_auc[t] = 0.1

    def get_confusion_metrics(self, filepath):
        self.model_tag_auc["sensitivity"] = 0
        self.model_tag_auc["specificity"] = 0
        self.model_tag_auc["precision"] = 0
        self.model_tag_auc["NPV(neg_pred_value)"] = 0
        self.model_tag_auc["accuracy"] = 0

        df = pd.read_csv(filepath)
        tags_with_targets = [
            t for t in df.columns.tolist() if f"{t}_target" in df.columns.tolist()
        ]

        for t in tags_with_targets:
            df_tag = df[df[f"{t}_target"] != -100]
            try:
                tn, fp, fn, tp = confusion_matrix(
                    df_tag[f"{t}_target"],
                    (df_tag[t] >= self.model_tag_auc[f"{t}_youden_index_cuttoff"]) * 1,
                ).ravel()

                self.model_tag_auc[f"{t}_sensitivity"] = tp / (tp + fn)
                self.model_tag_auc[f"{t}_specificity"] = tn / (tn + fp)
                self.model_tag_auc[f"{t}_precision"] = tp / (tp + fp)
                self.model_tag_auc[f"{t}_accuracy"] = (tn + tp) / (tn + tp + fn + fp)
                self.model_tag_auc[f"{t}_NPV"] = tn / (tn + fn)
                self.model_tag_auc[f"{t}_tp"] = tp
                self.model_tag_auc[f"{t}_tn"] = tn
                self.model_tag_auc[f"{t}_fp"] = fp
                self.model_tag_auc[f"{t}_fn"] = fn

                self.model_tag_auc["sensitivity"] += self.model_tag_auc[
                    f"{t}_sensitivity"
                ]
                self.model_tag_auc["specificity"] += self.model_tag_auc[
                    f"{t}_specificity"
                ]
                self.model_tag_auc["accuracy"] += self.model_tag_auc[f"{t}_accuracy"]
                self.model_tag_auc["precision"] += self.model_tag_auc[f"{t}_precision"]
                self.model_tag_auc["NPV(neg_pred_value)"] += self.model_tag_auc[
                    f"{t}_NPV"
                ]

            except Exception as e:
                print(e)
                print("=" * 50, f"error while cal confusion_metrics for {t}", "=" * 50)
                self.model_tag_auc[t] = 0.1

    def generate_confusion_metrics_for_all_models(self, user_filename="auc"):
        all_models_auc_scores = {}
        for model_path in self.model_list:
            self.model_tag_auc = {}
            self.get_paths_from_modelpath(model_path)
            print(f"Calculating it for {model_path}")
            self.get_auROC(self.auc_csv_fpath)
            self.get_auPRC(self.auc_csv_fpath)
            self.get_F1score(self.auc_csv_fpath)
            self.get_MCC(self.auc_csv_fpath)
            self.get_confusion_metrics(self.auc_csv_fpath)
            all_models_auc_scores[model_path] = self.model_tag_auc

        auc_csv = (
            pd.DataFrame.from_dict(all_models_auc_scores)
            .transpose()
            .reset_index()
            .rename(columns={"index": "model_name"})
        )
        auc_csv.sort_values(
            by="AUROC", ascending=False, na_position="last", inplace=True
        )
        auc_csv.to_csv(f"{self.results_folder}/{user_filename}_.csv", index=False)

    def get_paths_from_modelpath(self, modelpath):
        self.model_name = os.path.basename(modelpath).replace(
            "." + f'{modelpath.split(".")[-1]}', ""
        )
        self.model_folder = f"{self.checkpoint_dir}/results/{self.description}/{self.model_file}/{self.test_data_type}/{self.model_name}"
        self.results_folder = (
            f"{self.checkpoint_dir}/results/{self.description}/{self.model_file}/{self.test_data_type}"
        )
        self.auc_csv_fpath = os.path.join(self.model_folder, self.model_name + ".csv")

    def generate_iou_for_all_models(self):
        all_models_iou_scores = {}
        for model_path in self.model_list:
            self.get_paths_from_modelpath(model_path)

            all_models_iou_scores[model_path] = self.get_iou(self.model_folder)
            iou_csv = (
                pd.DataFrame.from_dict(all_models_iou_scores)
                .transpose()
                .reset_index()
                .rename(columns={"index": "model_name"})
            )
            self.iou_path = f"{self.results_folder}/iou.csv"
            iou_csv.to_csv(self.iou_path, index=False)

    def get_iou(self, model_folder):
        iou_dat = {}
        tags = [os.path.basename(t) for t in glob.glob(f"{model_folder}/*")]
        valid_tags = list({
            t for t in tags if any(len(glob.glob(f"{dir}/{t}/*")) > 0 for dir in self.test_annotation_dir)  ## ensure unique tags 
        })
        valid_tags = set(tags).intersection(set(valid_tags)) 
        print("iou_valid_tags: " , valid_tags)
        for t in valid_tags:
            ioud = iou_dirs(preds_dir=model_folder, gt_dirs=self.test_annotation_dir, tag=t)
            iou_dat[t] = ioud["iou"]
            iou_dat[f"{t}_th"] = ioud["iou_th"]
        return iou_dat

    def run_metrics(self):
        if "cls" in self.metric_type:
            print("calculating auroc ,auprc and confusion metrics")
            self.generate_confusion_metrics_for_all_models()

        if "seg" in self.metric_type:
            print("calculating iou")
            self.generate_iou_for_all_models()
