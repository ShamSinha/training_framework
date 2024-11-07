import numpy as np
import pandas as pd
from typing import List
import fastcore.all as fc
from functools import partial
from mdutils.mdutils import MdUtils

from voxdet.metrics.det_metrics import DetMetrics
from voxdet.metrics.sub_level_analysis import volume_callback, texture_callback, get_dataset_summary

from utils import get_concise_metrics, generate_recall_curves, get_source_df

class ReportGenerator:
    def __init__(self, file_name:str, title:str, root:str, dirs:List[str], meters:List[DetMetrics], subgroup_analysis:bool=False, source_analysis:bool=False):
        fc.store_attr()
        self.mdfile = MdUtils(file_name=self.file_name, title=self.title)
        self.bins = [[0, 113], [113, 268], [268, 1767], [1767, 14137], [14137, np.inf]]
        self.volume_bin_mapper = {"xsmall": 0, "small": 1, "medium": 2, "large": 3, "xlarge": 4}
        self.reset()

    def reset(self):
        self.recalls, self.num_nodules = [], []

    def __write_df_to_report(self, df:pd.DataFrame, title:str=None):
        """
        write given pandas dataframe to the report
        :arg title: spcified H2 title(heading)
        """
        if title: self.mdfile.write(f"\n## {title}\n")
        self.mdfile.insert_code(df.to_string(), language="bash")

    def __draw_recall_curves(self, subgroup_name:str):
        """
        draw recall curves for given subgroup analysis
        :arg subgroup_name: what subgroup are these recall curves for
        """
        img_name = generate_recall_curves(root=self.root, recalls=self.recalls, num_nodules=self.num_nodules, subgroup_name=subgroup_name)
        self.mdfile.write(f"\n\n![plot]({img_name})")

    def generate_dataset_summary(self, write:bool=False):
        """
        generates a summary of all datasets used for testing by #scans and #nodules per dataset
        :arg write: write dataset summary df to report
        """
        summary = get_dataset_summary(dirs=self.dirs, root=self.root)
        summary['all'] = {'num_scans': sum(item['num_scans'] for item in summary.values()), 'num_nodules': sum(item['num_nodules'] for item in summary.values())}
        self.summary_df = pd.DataFrame.from_dict(summary, orient='index')
        if write: self.__write_df_to_report(self.summary_df, title="Dataset Summary")

    def generate_source_analysis(self, write:bool=False):
        """
        performs source-wise analysis of metrics for all datasets used for testing
        :arg write: write source analysis df to report
        """
        self.generate_dataset_summary() # needs summary_df
        self.source_df = get_source_df(self.root, self.meters, self.summary_df)
        if write: self.__write_df_to_report(self.source_df, title="Source Analysis")

    def generate_metrics(self, write:bool=False):
        """
        computes metrics
        :arg write: write metrics df to report
        """
        self.metrics_df = get_concise_metrics(root=self.root, dirs=self.dirs, meters=self.meters)
        self.metrics_df['fpr'] = (self.metrics_df['fp']/self.summary_df['num_scans']).round(2)
        self.metrics_df.drop(['tp', 'fp', 'fn'], axis=1, inplace=True)
        if write: self.__write_df_to_report(self.metrics_df, title="Metrics")

    def generate_subgroup_metrics(self, callbacks, write:bool=False):
        """
        computes metrics for given subgroup(determined by the callback)
        :arg callbacks: support for volume and texture callbacks defined in sub_level_analysis
        :arg write: write subgroup df to report
        """
        self.subgroup_df = get_concise_metrics(root=self.root, dirs=self.dirs, meters=self.meters, callbacks=callbacks)
        self.recalls.append(self.subgroup_df.loc['all']['recall'])
        self.num_nodules.append(self.subgroup_df.loc['all']['tp']+self.subgroup_df.loc['all']['fn'])
        self.subgroup_df.drop(['tp', 'fp', 'fn'], axis=1, inplace=True)
        if write: self.__write_df_to_report(self.subgroup_df)

    def generate_report(self):
        """
        driver function to generate report
        """
        self.generate_dataset_summary(write=True)
        self.generate_metrics(write=True)
        if self.subgroup_analysis: self.__generate_subgroup_report()
        if self.source_analysis: self.generate_source_analysis()
        self.mdfile.create_md_file()

    def __generate_subgroup_report(self):
        self.__generate_volume_subgroup_report()
        self.__generate_texture_subgroup_report()

    def __generate_volume_subgroup_report(self):
        """
        computes metrics for all volume subgroups, writes it to the report and draws recall curves
        """
        self.mdfile.write("\n## Subgroup Analysis by Volume\n")
        self.reset()
        for key in self.volume_bin_mapper:
            bin_idx = self.volume_bin_mapper[key]
            self.mdfile.write(f"\n### {key} nodules: {self.bins[bin_idx]}")
            self.generate_subgroup_metrics(callbacks=partial(volume_callback, bins=self.bins, index=bin_idx), write=True)
        self.__draw_recall_curves(subgroup_name="volume")
    
    def __generate_texture_subgroup_report(self):
        """
        computes metrics for all texture subgroups, writes it to the report and draws recall curves
        """
        self.mdfile.write("\n## Subgroup Analysis by Texture\n")
        self.reset()
        for key in ["solid", "part solid", "non solid"]:
            self.mdfile.write(f"\n### {key} nodules")
            self.generate_subgroup_metrics(callbacks=partial(texture_callback, texture=key), write=True)
        self.__draw_recall_curves(subgroup_name="texture")
