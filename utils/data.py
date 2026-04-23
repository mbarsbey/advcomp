import torch
import numpy as np
from copy import deepcopy
import os
from PIL import Image
import pandas as pd
from .common import get_wrane_mask
from functools import partial
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets.vision import VisionDataset
from torchvision import datasets, transforms

class ScaleToRange:
    def __init__(self, input_min, input_max, output_min=0, output_max=1):
        self.input_min = input_min
        self.input_max = input_max
        self.output_min = output_min
        self.output_max = output_max

    def __call__(self, tensor):
        # Ensure tensor is a floating point type for the arithmetic operations
        tensor = torch.tensor(tensor, dtype=torch.float32)

        # Scale tensor in-place from [input_min, input_max] to [output_min, output_max]
        tensor.sub_(self.input_min).div_(self.input_max - self.input_min).mul_(self.output_max - self.output_min).add_(self.output_min)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(input_range=({self.input_min}, {self.input_max}), output_range=({self.output_min}, {self.output_max}))'

class Normalize:
    def __init__(self, mean, std):
        if len(mean)> 1:
            raise NotImplementedError
        self.mean = mean[0]
        self.std = std[0]

    def __call__(self, tensor):
        # Scale tensor from [input_min, input_max] to [output_min, output_max]
        return (tensor - self.mean) / self.std

    # def __repr__(self):
    #     return self.__class__.__name__ + f'(input_range=({self.input_min}, {self.input_max}), output_range=({self.output_min}, {self.output_max}))'

class ToLong:
    def __call__(self, tensor):
        return tensor.long()
class ToTensor:
    def __call__(self, array):
        return torch.tensor(array)


class FakeArgs():
  def __init__(
    self,
    dataset="mnist",
    batch_size_train=100,
    batch_size_eval=100,
    path="data",
    # data_scale=1,
    lr=0.1,
    criterion="NLL",
    device=None
    ):
    self.dataset = dataset
    self.batch_size_train = batch_size_train
    self.batch_size_eval = batch_size_eval
    self.path = path
    # self.data_scale = data_scale
    self.lr = lr
    self.criterion = criterion
    self.device = device
    self.momentum = 0.0
    self.wd = [0.0]
    self.lr_algorithm = "sgd"
    self.lr_scheduler = None

def convert_parity_to_fixed(data_loader, batch_size):
    xs, ys, bs = [], [], []
    for x, y, b in data_loader:
        xs.append(x), ys.append(y), bs.append(b)
    data, target, biased_target = torch.concatenate(xs), torch.concatenate(ys), torch.concatenate(bs)
    return DataLoader(
        dataset=Parity(X=data, y=target, b=biased_target, return_biased=True),
        batch_size=batch_size,
        shuffle=True
        )

def get_data(args=None, return_biased=False, abtrain=False, **kwargs):
    eval_loaders = {}
    
    if args.dataset in ['cifar100',]:
        data_class = 'CIFAR100'
        num_classes = 100
        input_dim = 32 * 32 * 3
        stats = {
            'mean': [0.5071, 0.4867, 0.4408] ,
            'std': [0.2675, 0.2565, 0.2761]
            }
    
    elif args.dataset in ["mnist"]:
        data_class = 'MNIST'
        num_classes = 10
        input_dim = 28 * 28
        stats = {'mean': [0.1307], 'std': [0.3081]}
    
    elif 'cifar10' in args.dataset:
        data_class = 'CIFAR10'
        num_classes = 10
        input_dim = 32 * 32 * 3
        if "wan2023" in args.dataset:
            stats = {
                'mean': [0., 0., 0.],
                'std': [1., 1., 1.]
                }
        else:
            stats = {
                'mean': [0.491, 0.482, 0.447],
                'std': [0.247, 0.243, 0.262]
                }
    elif 'svhn' in args.dataset:
        data_class = 'SVHN'
        num_classes = 10
        input_dim = 32 * 32 * 3
        stats = {
            'mean': [0.4377, 0.4438, 0.4728],
            'std': [0.1980, 0.2010, 0.1970]
            }
    else:
        raise ValueError("unknown dataset")

    trans = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        Normalize(**stats) if "dcase" in args.dataset else transforms.Normalize(**stats)
        ]

    
    
    # get tr and te data with the same normalization
    # no preprocessing for now
    data_aug = getattr(args, "data_augmentation", "none") != "none"
    if data_aug:
        assert args.dataset != "mnist"
        
        aug_trans = [transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),]
        
        if args.data_augmentation == "full":
            aug_trans += [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=15, # Rotates between -15 and +15 degrees
                        translate=(0.1, 0.1), # Translates up to 10% of image width/height
                        scale=(0.9, 1.1), # Scales between 90% and 110%
                        shear=10, # Shears by +/- 10 degrees
                        interpolation=transforms.InterpolationMode.BILINEAR)
            ]
        
        aug_trans = aug_trans + trans
    else:
        aug_trans = trans
    
    split = {"split": "test"} if "svhn" in args.dataset else {"train": False}
        
    te_data = getattr(datasets, data_class)(
        root=args.path,
        download=True,
        transform=transforms.Compose(trans),
        **split
    )

    tr_data = None
    val_data = None
    split = {"split": "train"} if "svhn" in args.dataset else {"train": True}
    if getattr(args, "val_criterion", None) and args.val_ratio > 0 and args.val_ratio < 1:
        
        tr_full_data = getattr(datasets, data_class)(
            root=args.path,
            download=True,
            transform=transforms.Compose(aug_trans),
            **split
        )

        tr_full_eval_data = getattr(datasets, data_class)(
            root=args.path,
            download=True,
            transform=transforms.Compose(trans),
            **split
        )

        val_full_data = getattr(datasets, data_class)(
            root=args.path,
            download=True,
            transform=transforms.Compose(trans),
            **split
        )

        dataset_length = len(tr_full_data)
        val_len = int(args.val_ratio * dataset_length)
        train_len = dataset_length - val_len

        generator = torch.Generator().manual_seed(args.seed)

        tr_indices_subset, val_indices_subset = torch.utils.data.random_split(
            val_full_data,
            [train_len, val_len],
            generator=generator
        )

        tr_data = Subset(tr_full_data, tr_indices_subset.indices)
        tr_eval_data = Subset(tr_full_eval_data, tr_indices_subset.indices)
        val_data = Subset(val_full_data, val_indices_subset.indices)
        val_data = IndexedDataset(val_data)
    else:
        tr_data = getattr(datasets, data_class)(
            root=args.path,
            download=True,
            transform=transforms.Compose(aug_trans),
            **split
        )
        tr_eval_data = getattr(datasets, data_class)(
            root=args.path,
            download=True,
            transform=transforms.Compose(trans),
            **split
        )
    tr_data, tr_eval_data, te_data = IndexedDataset(tr_data), IndexedDataset(tr_eval_data), IndexedDataset(te_data)
    
    try:
        tr_eval_data
    except:
        tr_eval_data = tr_data

        #     tra
    # get tr_loader for train/eval and te_loader for eval

    train_loader = DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_train,
        shuffle=True,
        )
    eval_loaders["train"] = DataLoader(
        dataset=tr_eval_data,
        batch_size=args.batch_size_eval,
        shuffle=True,
        )
    eval_loaders["test"] =  DataLoader(
        dataset=te_data,
        batch_size=args.batch_size_eval,
        shuffle=True,
        )
    if getattr(args, "val_criterion", ""):
        eval_loaders["val"] =  DataLoader(
        dataset=val_data,
        batch_size=args.batch_size_eval,
        shuffle=True,
        )
      
    return train_loader, eval_loaders, num_classes, input_dim


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        # If dataset is a Subset, check its base dataset
        base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset

        if hasattr(base, "data"):
            self.data = base.data
        if hasattr(base, "targets"):
            self.targets = base.targets
        if hasattr(base, "classes"):
            self.classes = base.classes
        if hasattr(base, "class_to_idx"):
            self.class_to_idx = base.class_to_idx
        if hasattr(base, "n_classes"):
            self.n_classes = base.n_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return idx, x, y