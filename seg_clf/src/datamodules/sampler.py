import numpy as np
import torch
from torch.utils.data import Sampler, BatchSampler, WeightedRandomSampler
from multiprocessing import Pool
from tqdm import tqdm
from loguru import logger

def make_weights_for_balanced_classes(labels, ignore_index=-100):
    unique_labels = np.unique(labels)
    nclasses = unique_labels[unique_labels != ignore_index].shape[0]

    count = [0] * nclasses
    
    # Counting each class while excluding the ignore index
    for item in labels:
        if item != ignore_index:
            count[item] += 1

    weight_per_class = [0.0] * nclasses
    N = float(sum(count))

    for i in range(nclasses):
        # Avoid division by zero if a class has no samples
        weight_per_class[i] = N / float(count[i]) if count[i] > 0 else 0.0
    
    logger.debug(weight_per_class)

    weight = [0] * len(labels)

    for idx, label in enumerate(labels):
        if label != ignore_index:
            weight[idx] = weight_per_class[label]
        else:
            # Assign zero weight to samples with the ignore index
            weight[idx] = 0

    return weight

def get_balanced_sampler_for_clf_task(train_data_list) :
        labels = [item['label'] for item in train_data_list]   
        logger.debug(len(labels))  
        weights = make_weights_for_balanced_classes(labels)
        sampler = WeightedRandomSampler(weights, len(train_data_list), replacement=True)
        return sampler

class CustomSampler(Sampler):
    def __init__(self, train_data_list, frac_per_class, num_samples):
        self.train_data_list = train_data_list
        self.frac_per_class = frac_per_class
        self._num_samples = num_samples
        self.positive_indices = {label: [] for label in ['ICH', 'IPH', 'SDH', 'SAH', 'EDH', 'IVH']}
        self.negative_indices_ich = []

        for i, data_dict in tqdm(enumerate(train_data_list), desc='Sampling progress', total=len(train_data_list)):
            try:
                if data_dict['ICH']:
                    for label in ['ICH', 'IPH', 'SDH', 'SAH', 'EDH', 'IVH']:
                        if data_dict[label]:
                            self.positive_indices[label].append(i)
                else:
                    self.negative_indices_ich.append(i)
            except: 
                continue

        # Determine the number of samples to draw
        self.num_samples_per_class = self.num_samples//2
                                  
        
    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.train_data_list)
        return self._num_samples

    def __iter__(self):
        positive_indices_per_label = {}                         
                                  
        for label in ['IPH', 'SDH', 'SAH', 'EDH', 'IVH'] :
            if int(self.num_samples_per_class*self.frac_per_class[label]) < len(self.positive_indices[label]) :
                size_per_label = int(self.num_samples_per_class*self.frac_per_class[label])
                positive_indices_per_label[label] = np.random.choice(self.positive_indices[label], size= size_per_label , replace=False)  
            else: 
                positive_indices_per_label[label] = np.random.choice(self.positive_indices[label], size=len(self.positive_indices[label]) , replace=False)
            
    
        positive_label_extra = {label: int(self.num_samples_per_class*self.frac_per_class[label]) - len(self.positive_indices[label])
                                  for label in ['IPH', 'SDH', 'SAH', 'EDH', 'IVH']}

        
        for label in ['IPH', 'SDH', 'SAH', 'EDH', 'IVH'] :
            if positive_label_extra[label] > 0 :
                positive_indices_per_label[label] = np.concatenate((positive_indices_per_label[label], np.random.choice(self.positive_indices[label], size=positive_label_extra[label], replace=True)))

        positive_ich_indices = [element for sublist in positive_indices_per_label.values() for element in sublist]

        np.random.shuffle(positive_ich_indices)
        # Sample negative indices for 'ICH' label
        negative_ich_indices = np.random.choice(self.negative_indices_ich, size = self.num_samples_per_class + self.num_samples%2, replace=False)

        indices = [x for pair in zip(positive_ich_indices, negative_ich_indices) for x in pair]
        np.random.shuffle(indices)
       
        return iter(indices)

    def __len__(self):
        return self.num_samples
    
class BalancedSampler(Sampler):
    def __init__(self, train_data_list, num_samples):
        print('started sampling')
        self.train_data_list = train_data_list
        self.num_samples = num_samples
        self.positive_indices = {label: [] for label in ['ICH', 'IPH', 'SDH', 'SAH', 'EDH', 'IVH']}
        self.negative_indices_ich = []

        for i, data_dict in tqdm(enumerate(train_data_list), desc='Sampling progress', total=len(train_data_list)):
            if data_dict['ICH']:
                for label in ['ICH', 'IPH', 'SDH', 'SAH', 'EDH', 'IVH']:
                    if data_dict[label]:
                        self.positive_indices[label].append(i)
            else:
                self.negative_indices_ich.append(i)

        # Calculate the minimum number of samples for each label
        min_samples_per_label = min(len(self.positive_indices[label]) for label in ['ICH', 'IPH', 'SDH', 'SAH', 'EDH', 'IVH'])
        min_samples_negative_ich = min(len(self.negative_indices_ich), min_samples_per_label)

        samples_per_label = [len(self.positive_indices[label]) for label in ['ICH', 'IPH', 'SDH', 'SAH', 'EDH', 'IVH']]

        # Save the minimum number of samples for each label
        self.min_samples_per_label = min_samples_per_label
        self.min_samples_negative_ich = min_samples_negative_ich

        # Check if num_samples is greater than the minimum samples
        if self.num_samples > min_samples_per_label:
            print("Warning: num_samples is greater than the minimum number of samples.")

    def __iter__(self):
        # Determine the number of samples to draw
        num_samples = min(self.num_samples, self.min_samples_per_label)

        # Sample positive indices for each label
        positive_label_indices = {label: np.random.choice(self.positive_indices[label], size=num_samples, replace=False)
                                  for label in ['ICH', 'IPH', 'SDH', 'SAH', 'EDH', 'IVH']}

        # Sample negative indices for 'ICH' label
        negative_ich_indices = np.random.choice(self.negative_indices_ich, size=num_samples*2, replace=False)

        # Concatenate the indices
        indices = np.concatenate(list(positive_label_indices.values()) + [negative_ich_indices])
        np.random.shuffle(indices)

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


class CustomSamplerV2(Sampler):
    def __init__(self, train_data_list, frac_per_class, num_samples, frac_without_mask):
        print('started sampling')
        self.train_data_list = train_data_list
        self.frac_per_class = frac_per_class
        self._num_samples = num_samples
        self.positive_indices_without_mask = {label: [] for label in ['ICH', 'IPH', 'SDH', 'SAH', 'EDH', 'IVH']}
        self.positive_indices_with_mask = {label: [] for label in ['ICH', 'IPH', 'SDH', 'SAH', 'EDH', 'IVH']}
        self.negative_indices_ich = []

        for i, data_dict in tqdm(enumerate(train_data_list), desc='Sampling progress', total=len(train_data_list)):
            try : 
                if data_dict['ICH']:
                    for label in ['ICH', 'IPH', 'SDH', 'SAH', 'EDH', 'IVH']:
                        if data_dict[label]:
                            if data_dict['mask'] == -100 :
                                self.positive_indices_without_mask[label].append(i)
                            else :
                                self.positive_indices_with_mask[label].append(i)
                else:
                    self.negative_indices_ich.append(i)
            except:
                continue

        # Determine the number of samples to draw
        self.num_samples_per_class = self.num_samples//2
        self.num_samples_per_class_without_mask = int(self.num_samples_per_class*frac_without_mask)
        self.num_samples_per_class_with_mask = self.num_samples_per_class - self.num_samples_per_class_without_mask
                                  
    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.train_data_list)
        return self._num_samples

    def __iter__(self):
        
        positive_mask_ich_indices = self.func(self.positive_indices_with_mask, self.num_samples_per_class_with_mask)
        positive_without_mask_ich_indices = self.func(self.positive_indices_without_mask, self.num_samples_per_class_without_mask)

        positive_ich_indices = positive_mask_ich_indices + positive_without_mask_ich_indices
        np.random.shuffle(positive_ich_indices)

        # Sample negative indices for 'ICH' label
        negative_ich_indices = np.random.choice(self.negative_indices_ich, size = self.num_samples_per_class + self.num_samples%2, replace=False)

        indices = [x for pair in zip(positive_ich_indices, negative_ich_indices) for x in pair]
        np.random.shuffle(indices)
       
        return iter(indices)
    
    def func(self, positive_indices , num_samples):
        positive_indices_per_label = {}                                           
        for label in ['IPH', 'SDH', 'SAH', 'EDH', 'IVH'] :
            if int(num_samples*self.frac_per_class[label]) < len(positive_indices[label]) :
                size_per_label = int(num_samples*self.frac_per_class[label])
                positive_indices_per_label[label] = np.random.choice(positive_indices[label], size= size_per_label , replace=False)  
            else: 
                positive_indices_per_label[label] = np.random.choice(positive_indices[label], size=len(positive_indices[label]) , replace=False)

            
        positive_label_extra = {label: int(num_samples*self.frac_per_class[label]) - len(positive_indices[label])
                                  for label in ['IPH', 'SDH', 'SAH', 'EDH', 'IVH']}

        for label in ['IPH', 'SDH', 'SAH', 'EDH', 'IVH'] :
            if positive_label_extra[label] > 0 :
                a= positive_indices_per_label[label]
                b = np.random.choice(positive_indices[label], size=positive_label_extra[label], replace=True)
                positive_indices_per_label[label] = np.concatenate((a , b))
            
        positive_ich_indices = [element for sublist in positive_indices_per_label.values() for element in sublist]
        np.random.shuffle(positive_ich_indices)

        return positive_ich_indices

    def __len__(self):
        return self.num_samples
    