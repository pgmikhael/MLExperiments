import torch
from torch.utils import data
from torchvision import datasets
from datasets.dataset_factory import RegisterDataset
import json
from collections import defaultdict
from tqdm import tqdm
import time
import os
from copy import copy
from datasets.abstract_dataset import Abstract_Dataset
from transformers.image import image_loader

@RegisterDataset("cifar10")
class Cifar10(object):
    """A pytorch Dataset for the Cifar data."""
    def __init__(self, args, img_transformers, tnsr_transformers, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        '''
        super(Cifar10, self).__init__()

        self.split_group = split_group
        self.args = args
        self.image_loader = image_loader(img_transformers, tnsr_transformers, args)
        self.dataset = self.create_dataset(split_group)
 
    
    def create_dataset(self, split_group):
        
        if split_group == 'train':
            dataset = datasets.CIFAR10('/dev/shm/cifar10', train=True, download=True)

        else:
            cifar10_test = datasets.CIFAR10('/dev/shm/cifar10', train=False, download=True)
            if split_group == 'dev':
                dataset = [cifar10_test[i] for i in range(len(cifar10_test) // 2)]
            elif split_group == 'test':
                dataset = [cifar10_test[i] for i in range(len(cifar10_test) // 2, len(cifar10_test))]

            else:
                raise Exception('Split group must be in ["train"|"dev"|"test"].')
        
        return dataset

    @staticmethod
    def set_args(args):
        args.num_classes = 10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = self.image_loader.composed_all_transformers(x)
        
        item = {
            'x': x,
            'y': y
        }

        return item

    @property
    def METADATA_FILENAME(self):
        return 'cifar10'

@RegisterDataset("cifar100")
class Cifar100(object):
    """A pytorch Dataset for the Cifar data."""

    def __init__(self, args, img_transformers, tnsr_transformers, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        '''
        super(Cifar100, self).__init__()

        self.split_group = split_group
        self.args = args
        self.image_loader = image_loader(img_transformers, tnsr_transformers, args)
        self.dataset = self.create_dataset(split_group)
    
    def create_dataset(self, split_group): 
        if split_group == 'train':
            dataset = datasets.CIFAR10('/dev/shm/cifar10', train=True, download=True)

        else:
            cifar10_test = datasets.CIFAR10('/dev/shm/cifar10', train=False, download=True)
            if split_group == 'dev':
                dataset = [cifar10_test[i] for i in range(len(cifar10_test) // 2)]
            elif split_group == 'test':
                dataset = [cifar10_test[i] for i in range(len(cifar10_test) // 2, len(cifar10_test))]

            else:
                raise Exception('Split group must be in ["train"|"dev"|"test"].')
        
        return dataset

    @staticmethod
    def set_args(args):
        args.num_classes = 10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = self.image_loader.composed_all_transformers(x)
        
        item = {
            'x': x,
            'y': y
        }

        return item
    
    @property
    def METADATA_FILENAME(self):
        return 'cifar100'