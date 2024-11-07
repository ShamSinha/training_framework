from cxr_training.data.data_utils.data_processing import ImageLabel
import torch.utils.data as data
import pytorch_lightning as pl
from cxr_training.nnmodule.models.utils import get_class_from_str
from cxr_training.data.data_utils.distributed_proxy_sampler import (
    DistributedProxySampler,
)


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.train_samples = args.trainer.train_samples
        self.validation_samples = args.trainer.validation_samples
        self.batch_size = args.trainer.batch_size
        self.num_process = args.trainer.num_workers
        self.strategy = args.trainer.strategy
        self.sampler = args.params.sampler
        self.dataset = get_class_from_str(args.params.dataset_type)
        print(f"\n process count is {self.num_process} \n")
        self._prepare_data()
        self._setup()

    def _prepare_data(self):
        """private method that Define steps that should be done on only parent GPU (rank:0) of each node,
        like download/processing data."""
        self.metadata_df = ImageLabel(self.args).get_data()
        print(f"no of images after inner join is {len(self.metadata_df)}")

    def _setup(self):
        """
        private method that has  data operations you might want to perform on every GPU. Use setup to do things like:
        count number of classes, build vocabulary,perform train/val/test splits ,apply transforms
        (defined explicitly in your datamodule)
        etc…
        """
        self.train_df = self.metadata_df[self.metadata_df["type"] == "train"]
        self.val_df = self.metadata_df[self.metadata_df["type"] == "val"]

    def create_sampler(
        self, mode, sampler_type, data_frame, num_samples, replacement_type=True
    ):

        if f"{mode}_random" in sampler_type:
            print(f"using random for {mode}", "\n \n")
            return data.RandomSampler(
                data_source=data_frame,
                num_samples=num_samples,
                replacement=replacement_type,
            )

        elif f"{mode}_weighted" in sampler_type:
            print(f"using weighted for {mode}", "\n \n")
            return data.WeightedRandomSampler(
                weights=data_frame["final_sample_wts"],
                num_samples=num_samples,
                replacement=replacement_type,
            )

    def train_dataloader(self):
        print("train dataloader")
        train_ds = self.dataset(self.args, self.train_df, mode="train")

        self.image_train_sampler = self.create_sampler(
            "train",
            self.sampler,
            self.train_df,
            self.train_samples,
            replacement_type=True,
        )
        print("Length of sampled images : " ,len(self.image_train_sampler))
        if "ddp" in self.strategy:
            print("Using Train DistributedProxySampler")
            self.image_train_sampler = DistributedProxySampler(self.image_train_sampler)

        train_dl = data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            sampler=self.image_train_sampler,
            num_workers=self.num_process,
            pin_memory=True,
            persistent_workers=True,
        )

        return train_dl

    def val_dataloader(self):
        print("val dataloader")
        val_ds = self.dataset(self.args, self.val_df, mode="val")

        self.image_val_sampler = self.create_sampler(
            "val",
            self.sampler,
            self.val_df,
            self.validation_samples,
            replacement_type=False,
        )

        if "ddp" in self.strategy:
            print("Using Val DistributedProxySampler")
            self.image_val_sampler = DistributedProxySampler(self.image_val_sampler)

        val_dl = data.DataLoader(
            val_ds,
            batch_size=self.batch_size,
            sampler=self.image_val_sampler,
            shuffle=False,
            num_workers=self.num_process,
            persistent_workers=True,
        )
        return val_dl

    def get_data_df(self):
        return self.metadata_df

    def get_train_df(self):
        return self.train_df

    def get_val_df(self):
        return self.val_df

    def get_train_sampler(self):
        return self.image_train_sampler

    def get_val_sampler(self):
        return self.image_val_sampler
