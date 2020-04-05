import torch
from torch.utils import data
from collections import defaultdict
from tqdm import tqdm
import pdb
import json
import time

NO_DATASET_ERR = "Dataset {} not in DATASET_REGISTRY! Available datasets are {}"

DATASET_REGISTRY = {}

def RegisterDataset(dataset_name):
    """Registers a dataset."""

    def decorator(f):
        DATASET_REGISTRY[dataset_name] = f
        return f

    return decorator


def get_dataset_class(args):
    if args.dataset not in DATASET_REGISTRY:
        raise Exception(
            NO_DATASET_ERR.format(args.dataset, DATASET_REGISTRY.keys()))

    return DATASET_REGISTRY[args.dataset]


def get_dataset(args, img_transformers, tnsr_transformers, split_group):
    dataset_class = get_dataset_class(args)
    dataset  =  dataset_class(args, img_transformers, tnsr_transformers, split_group)
    
    return dataset

