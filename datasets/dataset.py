import torch
from torch.utils import data
from datasets.dataset_factory import RegisterDataset
import json
from collections import defaultdict
from tqdm import tqdm
import time
import os
from copy import copy
from datasets.abstract_dataset import Abstract_Dataset
from helpers.classes import Ddict

METADATA_FILENAMES = {"prediction": "dataset.json"}

@RegisterDataset("dataset")
class Dataset(Abstract_Dataset):

    def create_dataset(self, split_group):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        """
        dataset = []
        for user_row in tqdm(self.metadata_json):
            user_id, split, posts = user_row['user_id'], user_row['split'], user_row['posts']

            if not split == split_group:
                continue

            for i, post in enumerate(posts):
                if post['is_video']:
                    continue

                y = self.get_label(post, self.args.task) 

                for img in post['imgs']:
                    if img is not None and os.path.exists(os.path.join(self.args.img_dir, img)):
                        dataset.append({
                            'user_id': user_id,
                            'post_id': post['post_id'],
                            'path': img,
                            'y': y,
                            'tags': post['tags'],
                            'followers':user_row['followers']
                        })

        return dataset


    def get_label(self, post, task):
        pass

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAMES[self.task]

    @staticmethod
    def set_args(args):
        args.num_classes = 1

    @property
    def task(self):
        return "prediction"