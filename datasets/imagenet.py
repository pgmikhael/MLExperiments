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


@RegisterDataset("imagenet")
class Imagenet(Abstract_Dataset):
    """A pytorch Dataset for the ImageNet data."""
    
    def create_dataset(self, split_group):
        self.split_group = split_group
        if self.split_group == 'train':
            self.dataset = datasets.ImageNet('imagenet',
                                          train=True,
                                          download=True)
        else:
            imagenet_test = datasets.ImageNet('imagenet',
                                        train=False,
                                        download=True)
            if self.split_group == 'dev':
                self.dataset = [imagenet_test[i] for i in range(len(imagenet_test) // 2)]
            elif self.split_group == 'test':
                self.dataset = [mnist_timagenet_testest[i] for i in range(len(imagenet_test) // 2, len(imagenet_test))]
            else:
                raise Exception('Split group must be in ["train"|"dev"|"test"].')

    @staticmethod
    def set_args(args):
        args.num_classes = 1000

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