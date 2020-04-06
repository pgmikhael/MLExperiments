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


@RegisterDataset("cifar10")
class Cifar10(Abstract_Dataset):
    """A pytorch Dataset for the ImageNet data."""
    
    def create_dataset(self, split_group):
        self.split_group = split_group
        if self.split_group == 'train':

            self.dataset = datasets.CIFAR10('cifar10', train=True, download=True)

        else:
            cifar10_test = datasets.CIFAR10('cifar10', train=False, download=True)
            if self.split_group == 'dev':
                self.dataset = [cifar10_test[i] for i in range(len(cifar10_test) // 2)]
            elif self.split_group == 'test':
                self.dataset = [cifar10_test[i] for i in range(len(cifar10_test) // 2, len(cifar10_test))]
            else:
                raise Exception('Split group must be in ["train"|"dev"|"test"].')

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

@RegisterDataset("cifar100")
class Cifar100(Abstract_Dataset):
    """A pytorch Dataset for the ImageNet data."""
    
    def create_dataset(self, split_group):
        self.split_group = split_group
        if self.split_group == 'train':

            self.dataset = datasets.CIFAR100('cifar100', train=True, download=True)

        else:
            cifar100_test = datasets.CIFAR100('cifar100',train=False, download=True)
            if self.split_group == 'dev':
                self.dataset = [cifar100_test[i] for i in range(len(cifar100_test) // 2)]
            elif self.split_group == 'test':
                self.dataset = [cifar100_test[i] for i in range(len(cifar100_test) // 2, len(cifar100_test))]
            else:
                raise Exception('Split group must be in ["train"|"dev"|"test"].')

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