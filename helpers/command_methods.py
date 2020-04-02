import torch
import os.path
from datasets.dataset_factory import get_dataset
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

def get_dataset_stats(args):
    args = copy.deepcopy(args)
    img_transformers = ['scale_2d']
    tnsr_transformers = ['force_num_chan']
    args.computing_stats = True
    train = get_dataset(args, img_transformers, tnsr_transformers, 'train')

    data_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False)

    means, stds = {0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}
    # num_samples = 0
    for batch in tqdm(data_loader):
        tensor = batch['x']

        if args.cuda:
            tensor = tensor.cuda()

        for channel in range(3):
            tensor_chan = tensor[:, channel]
            means[channel].append(torch.mean(tensor_chan))
            stds[channel].append(torch.std(tensor_chan))

    means = [torch.mean(torch.Tensor(means[channel])) for channel in range(3)]
    stds = [torch.mean(torch.Tensor(stds[channel])) for channel in range(3)]

    return means, stds 

