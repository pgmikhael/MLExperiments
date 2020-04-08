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
from transformers.image import image_loader


@RegisterDataset("imagenet")
class Imagenet(object):
    """A pytorch Dataset for the ImageNet data."""
    def __init__(self, args, img_transformers, tnsr_transformers, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        '''
        super(Imagenet, self).__init__()

        self.split_group = split_group
        self.args = args
        self.image_loader = image_loader(img_transformers, tnsr_transformers, args)
        self.dataset = self.create_dataset(split_group)
    
    def create_dataset(self, split_group):
        
        if split_group == 'train':
            dataset = datasets.ImageNet('/dev/shm/imagenet', train=True, download=True)
        else:
            imagenet_test = datasets.ImageNet('/dev/shm/imagenet', train=False, download=True)
            if split_group == 'dev':
                dataset = [imagenet_test[i] for i in range(len(imagenet_test) // 2)]
            elif split_group == 'test':
                dataset = [imagenet_test[i] for i in range(len(imagenet_test) // 2, len(imagenet_test))]
            else:
                raise Exception('Split group must be in ["train"|"dev"|"test"].')
        
        return dataset

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

    @property
    def METADATA_FILENAME(self):
        return 'imagenet'