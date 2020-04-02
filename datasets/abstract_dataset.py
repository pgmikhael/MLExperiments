import numpy as np
import pickle
from abc import ABCMeta, abstractmethod
import torch
from torch.utils import data
import os
import warnings
import json
import traceback
from collections import Counter
from transformers.image import image_loader
from scipy.stats import entropy
import pdb

METAFILE_NOTFOUND_ERR = "Metadata file {} could not be parsed! Exception: {}!"
LOAD_FAIL_MSG = "Failed to load image: {}\nException: {}"


DATASET_ITEM_KEYS = ['y']

class Abstract_Dataset(data.Dataset):
    """
    Abstract Object for all Onco Datasets. All datasets have some metadata
    property associated with them, a create_dataset method, a task, and a check
    label and get label function.
    """
    __metaclass__ = ABCMeta

    def __init__(self, args, img_transformers, tnsr_transformers, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        '''
        super(Abstract_Dataset, self).__init__()

        args.metadata_path = os.path.join(args.data_dir, self.METADATA_FILENAME)

        self.split_group = split_group
        self.args = args
        self.image_loader = image_loader(img_transformers, tnsr_transformers, args)
        try:
            self.metadata_json = json.load(open(args.metadata_path, 'r'))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.metadata_path, e))

        self.dataset = self.create_dataset(split_group)

        if len(self.dataset) == 0:
            return

        label_dist = [d['y'] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1./ len(label_counts)
        label_weights = { label: weight_per_label/count for label, count in label_counts.items()}
        # if args.class_bal:
        if self.task is not 'regression':
            print("Label weights are {}".format(label_weights))
            self.weights = [ label_weights[d['y']] for d in self.dataset]

    @property
    @abstractmethod
    def task(self):
        pass

    @property
    @abstractmethod
    def METADATA_FILENAME(self):
        pass

    @abstractmethod
    def check_label(self, row):
        '''
        Return True if the row contains a valid label for the task
        :row: - metadata row
        '''
        pass

    @abstractmethod
    def get_label(self, row):
        '''
        Get task specific label for a given metadata row
        :row: - metadata row with contains label information
        '''
        pass

    def get_summary_statement(self, dataset, split_group):
        '''
        Return summary statement
        '''
        return ""

    @abstractmethod
    def create_dataset(self, split_group):
        """
        Creating the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].

        """
        pass


    @staticmethod
    def set_args(args):
        """Sets any args particular to the dataset."""
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.get_image_item(index)


    def get_image_item(self, index):
        sample = self.dataset[index]

        ''' Region annotation for each image. Dict for single image,
            list of dict for multi-image
        '''
        try:
            x = self.image_loader.get_image(os.path.join(self.args.img_dir, sample['path']))

            item = {
                'x': x,
                'path': os.path.join(self.args.img_dir, sample['path']),
                'y': sample['y']
            }

            for key in DATASET_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            return item

        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample['path'], traceback.print_exc()))


