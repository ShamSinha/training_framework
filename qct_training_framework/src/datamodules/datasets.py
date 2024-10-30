# Author: Souvik, 2023
from functools import partial
from multiprocessing import Pool
from typing import Callable, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

__all__ = ["TFDataset", "SiameseDataset"]


class TFDataset(Dataset):
    def __init__(self, data: Sequence, phase: str, transform: Optional[Callable] = None) -> None:
        """Base dataset class for all tasks."""
        super().__init__()
        self.data = data
        self.transform = transform
        self.phase = phase

    def __getitem__(self, idx):
        # return should be a dictionary
        return self.transform(self.data[idx])

    def __len__(self):
        return len(self.data)


class SiameseDataset(TFDataset):
    def __init__(
        self,
        data: Sequence,
        phase: str,
        transform: Optional[Callable] = None,
        seed: int = 24,
        label_key: str = "label",
        image_key: str = "image",
    ) -> None:
        """Siamse Dataset for training and testing.

        Args:
            data: Sequence of data dictionaries
            phase: train, val or test
            transform: Transform to be applied on the data
            seed: Seed for random state
            label_key: Key for label in data dictionary
        """
        super().__init__(data=data, phase=phase, transform=transform)
        self.seed = seed
        self.image_key = image_key
        self.label_key = label_key

        self.random_state = np.random.RandomState(seed)
        labels = np.array([data[label_key] for data in self.data])
        self.unique_labels = np.unique(labels)
        self.label_to_indices = {
            label: np.where(labels == label)[0] for label in self.unique_labels
        }
        if self.phase != "train":
            # Create fix triplets for validation and testing
            with Pool() as pool:
                # generate triplets in parallel
                func = partial(self.generate_triplet, labels=labels)
                triplets = pool.map(func, range(len(self.data)))
            self.triplets = triplets

    def generate_triplet(self, i, labels):
        pos_idx = self.random_state.choice(self.label_to_indices[labels[i]])
        neg_label = self.random_state.choice(np.setdiff1d(self.unique_labels, [labels[i]]))
        neg_idx = self.random_state.choice(self.label_to_indices[neg_label])
        return [i, pos_idx, neg_idx]

    def __getitem__(self, idx):
        random_state = np.random.RandomState(self.seed)
        if self.phase == "train":
            data = self.data[idx]
            label = data[self.label_key]
            negative_index_cls = random_state.choice(np.setdiff1d(self.unique_labels, [label]))
            negative_data_dict = self.data[
                random_state.choice(self.label_to_indices[negative_index_cls])
            ]
            positive_data_dict = self.data[random_state.choice(self.label_to_indices[label])]
        else:
            anchor_idx, positive_idx, negative_idx = self.triplets[idx]
            data, positive_data_dict, negative_data_dict = (
                self.data[anchor_idx],
                self.data[positive_idx],
                self.data[negative_idx],
            )
        anchor_dict, positive_dict, negative_dict = (
            self.transform(data),
            self.transform(positive_data_dict),
            self.transform(negative_data_dict),
        )
        data = {
            "anchor": anchor_dict[self.image_key],
            "positive": positive_dict[self.image_key],
            "negative": negative_dict[self.image_key],
            f"{self.label_key}": anchor_dict[self.label_key],
        }

        return data
