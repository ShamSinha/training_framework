import math
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


def filter_by_src(df: pd.DataFrame, srcs: List[str], src_column: str, phase: str):
    """
    Filter a dataframe by sources mentioned.
    Args:
    -----
    df: pd.DataFrame to filter
    srcs: which sources should be included in the dataset
    src_column: column name in the dataframe which have sources information
    phase: training phase, train/val/test. Only used for logging purposes.
    """
    old_sample_size = df.shape[0]

    if srcs is not None and len(srcs) > 0:
        logger.info(f"{phase}: Filtering based on srcs: {srcs}")
        df = df[df[src_column].isin(srcs)] if len(srcs) > 0 and src_column in df else df

        logger.info(
            f"Sample size on src based filtering before: {old_sample_size}, after: {df.shape[0]}"
        )
    else:
        logger.info(f"No Src is mentioned for {phase}. Skipping src based filter.")

    return df


def get_bin_indices(bins: List, data: List, return_edge: bool = True):
    """
    Given a bins ([10, 20, 30, 40]) and a list of datapoints [2, 3, 10, 15, 25]
    returns a list of same length as the datapoints with the bin information
    ([0,0,0,1,2])
    Note: Max value of the data should not be greater than the max bin value and
    Args:
    -----
    bins: all the bins
    data: data as list
    return_edge: if true returns the bin edge instead of index
    """
    bin_indices = []
    for item in data:
        if item is None or math.isnan(item):
            bin_indices.append(None)
            continue
        for i, bin_edge in enumerate(bins):
            if item <= bin_edge:
                bin_indices.append(bin_edge if return_edge else i)
                break
    return bin_indices


def get_mappings_indices(mappings: Dict, meta_value: Optional[str]):
    """
    Given a characteristics mapping get the class id.
    Args:
    -----
    mappings: Mappings of a specific characteristics.
    Return:
    Index of the class of the meta_value.
    Eg:
    Texture: (mappings)
        Solid: texture_solid
        solid: texture_solid
        solid/mixed: texture_part_solid
    meta_value: solid
    Return: 1
    """
    values = sorted(list(set(list(mappings.values()))))
    return values.index(mappings[meta_value]) if meta_value is not None else -1


def extract_metadata_with_cfg(meta_value: Any, meta_name: Any, meta_cfg: Optional[Dict]):
    """Extract metadata value for subgroup analysis."""
    if meta_value is None:
        return -1
    if meta_cfg is None:
        return meta_value
    meta_cfg = meta_cfg[meta_name]
    if meta_cfg["type"] == "bin":
        bins = meta_cfg["bins"]
        return get_bin_indices(bins, [meta_value], return_edge=True)[0]
    elif meta_cfg["type"] == "mapping":
        mappings = meta_cfg["mappings"]
        return mappings[meta_value] if meta_value != "-1" else -1
    else:
        raise NotImplementedError(f"{meta_cfg['type']} is not implemented.")


def dataset_oversample(
    oversample_ratios: Optional[Dict], data_dict: List[Dict], label_key: str = "label"
):
    """Oversample datasets.

    Stacks data points of a class multiple number times based on the cfg.
    Args:
    -----
    oversample_ratios: oversample ratios to use
    data_dict: dataset in List[{Dict}] each element in the dataset is a Dict.
    label_key: Key name in the dict to extract the label.
    """
    if oversample_ratios is not None:
        logger.info(f"Using oversample. Ratios: {oversample_ratios}")
        data_dict_per_class = {}
        for data in data_dict:
            class_idx = data[label_key]
            if class_idx in data_dict_per_class:
                data_dict_per_class[class_idx].append(data)
            else:
                data_dict_per_class[class_idx] = [data]
        assert len(list(data_dict_per_class.keys())) == len(list(oversample_ratios.keys()))
        final_data = []
        for class_idx, ratio in oversample_ratios.items():
            final_data.extend(data_dict_per_class[class_idx] * ratio)
        return final_data
    return data_dict
