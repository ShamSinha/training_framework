from copy import deepcopy
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig

from . import *  # noqa
from .utils import dataset_oversample


def _extract_datasets(datasets_cfg: Optional[Union[DictConfig, List[DictConfig]]], phase: str, **kwargs):
    if not datasets_cfg :
        data_extract_func = eval(kwargs.pop("data_extract_func"))
        return data_extract_func(**kwargs , phase=phase)
    if isinstance(datasets_cfg, DictConfig):
        datasets_cfg = [datasets_cfg]
    dataset = []
    for dataset_cfg in datasets_cfg:
        dataset_cfg = deepcopy(dataset_cfg)
        data_extract_func = eval(kwargs.pop("data_extract_func") if "data_extract_func" in kwargs else dataset_cfg.pop("data_extract_func")) 
        dataset.extend(data_extract_func(**dataset_cfg, phase=phase, **kwargs))
    return dataset


def extract_dataset(
    train: Optional[Union[DictConfig, List[DictConfig]]] = None,
    val: Optional[Union[DictConfig, List[DictConfig]]]= None,
    test: Optional[Union[DictConfig, List[DictConfig]]] = None,
    oversample_ratios: Optional[Dict] = None,
    label_key: str = "label",
    **kwargs
):
    train_data_dict = _extract_datasets(train, phase="train", **kwargs)
    if oversample_ratios is not None:
        train_data_dict = dataset_oversample(
            oversample_ratios, train_data_dict, label_key=label_key
        )
    val_data_dict = _extract_datasets(val, phase="val", **kwargs)
    test_data_dict = _extract_datasets(test, phase="test", **kwargs)

    # check datasets are not empty
    assert len(train_data_dict) > 0, "train dataset is empty"
    assert len(val_data_dict) > 0, "val dataset is empty"
    assert len(test_data_dict) > 0, "test dataset is empty"

    return train_data_dict, val_data_dict, test_data_dict
