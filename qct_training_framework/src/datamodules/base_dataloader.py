from typing import Any, Dict

import lightning.pytorch as pl
from loguru import logger
from monai.data import Dataset
from monai.transforms import PadListDataCollate
from torch.utils.data import DataLoader, WeightedRandomSampler
from .sampler import BalancedSampler, CustomSampler, CustomSamplerV2, get_balanced_sampler_for_clf_task


from .load_data.datasets import extract_dataset
from time import time
import os
from tqdm.auto import tqdm

class MetaClsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_cfg: Dict[str, Any],
        transforms: Dict,
        dataloader_cfg: Dict[str, Any],
        **kwargs,
    ) -> None:
        """Meta Class lightning data module.

        Check the cfg file for description of the class.
        """
        super().__init__()
        self.data_cfg = data_cfg
        self.dataloader_cfg = dataloader_cfg
        self.transforms = transforms
        self.setup()
        self.logger = logger
        # self.logger.add(os.path.join("/home/users/shubham.kumar/projects/qct_training_framework", "test_optimal_num_workers_num_batch_size.log"), level="DEBUG")

    def setup(self, stage=None):
        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage}  not implemented.")
        train_data_list, val_data_list, test_data_list = extract_dataset(
            **self.data_cfg
        )
        self.train_data_list = train_data_list
        test_transforms = getattr(
            self.transforms, "test_transforms", self.transforms.val_transforms
        )
        if stage == "fit" or stage is None:
            self.train_ds = (
                Dataset(train_data_list, transform=self.transforms.train_transforms)
                if getattr(self, "train_ds", None) is None
                else self.train_ds
            )
            self.val_ds = (
                Dataset(val_data_list, transform=self.transforms.val_transforms)
                if getattr(self, "val_ds", None) is None
                else self.val_ds
            )
            logger.info(f"Train set Length: {len(self.train_ds)}")
            logger.info(f"Val set Length: {len(self.val_ds)}")
        if stage == "test" or stage is None:
            self.test_ds = (
                Dataset(test_data_list, transform=test_transforms)
                if getattr(self, "test_ds", None) is None
                else self.test_ds
            )
            logger.info(f"Test set Length: {len(self.test_ds)}")

    def train_dataloader(self):
        if self.dataloader_cfg.sample:
            train_sampler = CustomSampler(
                self.train_data_list,
                frac_per_class=self.dataloader_cfg.frac_per_class,
                num_samples=self.dataloader_cfg.num_samples,
            )
            return DataLoader(
                    self.train_ds,
                    batch_size=self.dataloader_cfg.batch_size,
                    sampler=train_sampler,
                    num_workers=self.dataloader_cfg.num_workers,
                    drop_last=True,
                    pin_memory=True,
                )
        if self.dataloader_cfg.sample_fusion:
            train_sampler = CustomSamplerV2(
                self.train_data_list,
                frac_per_class=self.dataloader_cfg.frac_per_class,
                num_samples=self.dataloader_cfg.num_samples,
                frac_without_mask= self.dataloader_cfg.frac_without_mask
            )
            return DataLoader(
                    self.train_ds,
                    batch_size=self.dataloader_cfg.batch_size,
                    sampler=train_sampler,
                    num_workers=self.dataloader_cfg.num_workers,
                    drop_last=True,
                    pin_memory=True,
                )
            
            # self.logger.debug(f"data_size: {len(self.train_ds)}")

            # for batch_size in [8,12,16,24,32] : 
            #     for num_workers in tqdm(range(4, 64, 4)):  
            #         train_dataloader = DataLoader(
            #             self.train_ds,
            #             batch_size=batch_size,
            #             # sampler=train_sampler,
            #             shuffle=True,
            #             num_workers=num_workers,
            #             drop_last=True,
            #             pin_memory=True,
            #         )
            #         start = time()
            #         for i, data in tqdm(enumerate(train_dataloader, 0)):
            #             pass
            #         end = time()
            #         self.logger.debug("Finish with:{} second, batch_size = {} ,num_workers={} ".format(end - start, batch_size ,num_workers))

        if self.dataloader_cfg.balanced_sample:
            sampler = get_balanced_sampler_for_clf_task(self.train_data_list)
            return DataLoader(
                self.train_ds,
                batch_size=self.dataloader_cfg.batch_size,
                sampler= sampler,
                num_workers=self.dataloader_cfg.num_workers,
                drop_last=True,
                pin_memory=True,
            )

        else:
            return DataLoader(
                self.train_ds,
                batch_size=self.dataloader_cfg.batch_size,
                shuffle=True,
                num_workers=self.dataloader_cfg.num_workers,
                drop_last=True,
                pin_memory=True,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.dataloader_cfg.batch_size,
            shuffle=False,
            num_workers=self.dataloader_cfg.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.dataloader_cfg.batch_size,
            shuffle=False,
            num_workers=self.dataloader_cfg.num_workers,
            pin_memory=True,
        )
