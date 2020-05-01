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

@RegisterDataset("mnist")
class MNIST_Dataset(Abstract_Dataset):
    """A pytorch Dataset for the MNIST data."""
    
    def create_dataset(self, split_group):
        self.split_group = split_group
        if self.split_group == 'train':
            self.dataset = datasets.MNIST('mnist',
                                          train=True,
                                          download=True)
        else:
            mnist_test = datasets.MNIST('mnist',
                                        train=False,
                                        download=True)
            if self.split_group == 'dev':
                self.dataset = [mnist_test[i] for i in range(len(mnist_test) // 2)]
            elif self.split_group == 'test':
                self.dataset = [mnist_test[i] for i in range(len(mnist_test) // 2, len(mnist_test))]
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

    @property
    def METADATA_FILENAME(self):
        return 'mnist'
