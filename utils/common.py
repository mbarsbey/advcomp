import torch
import numpy as np
import pickle, logging, json
import json
import os
import pandas as pd
from datetime import datetime, timedelta
import shutil
from distutils.dir_util import copy_tree
from collections.abc import Iterable
import psutil, pynvml
import torch.nn as nn

# from spgl1 import spg_bp

EPS = {"Linf": 8/255, "L2": .5}
STATS = {"cifar10":{'mean': [0.491, 0.482, 0.447],'std': [0.247, 0.243, 0.262]}, "cifar100":{'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]}, "mnist": {'mean': [0.1307],'std': [0.3081]}, "bmnist": {'mean': [0.1307],'std': [0.3081]}, "svhn": {'mean': [0.4377, 0.4438, 0.4728], 'std': [0.1980, 0.2010, 0.1970]}}
AVAILABLE_PARAMS = (2, 4)

device = torch.device('cuda')
logger = logging.getLogger(); logger.setLevel(logging.CRITICAL)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

def find_m(N, m=0):
    if m == 0:
        m = int(np.sqrt(N)); assert N < 3e7+1
    while N % m != 0:
       m -= 1
    return m
    
def get_layers(model, dream_team, as_numpy=False):
    layers = [layer for layer in model.parameters()]
    if dream_team:
        layers[-1] = layers[-1].T
    if as_numpy:
        return [layer.data.cpu().numpy() for layer in layers]
    return layers

def get_num_params(model):
    return sum([np.prod(layer.shape) for layer in model.parameters()])

def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)    
        
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)    

def clean_folder(folder):
    #return folder.replace("results/","").replace("/", "")
    return folder.split("/")[-2]

def get_layer_to_numpy(p):
    with torch.no_grad():
        a = p.to("cpu").numpy()
    return a

def get_folder_idx(orig_folder_length, k, T):
    no_exps = orig_folder_length // T
    start = (k-1) * no_exps
    end = k  * no_exps if k < T else orig_folder_length
    return no_exps, start, end

def get_results_part_string(eparallel, no_samples):
    results_part_string = "" if eparallel == "_1_1" else "_part" + eparallel
    if results_part_string == "":
        if no_samples != -1:
            results_part_string = f"_{no_samples}_samples"
    return results_part_string

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def get_model_info(m, dream_team, x_type):
    model_file_name = "avg_net.pyT" if x_type == "x_mc" else "net.pyT"
    model = torch.load(m + model_file_name,map_location='cpu')
    args = torch.load(m + "args.info",map_location='cpu')
    model_name = args.model.upper()
    dataset_name = args.dataset.upper()
    layers = get_layers(model, dream_team=dream_team)
    no_layers = len(layers)
    return model, args, model_name, dataset_name, layers, no_layers

def get_model_info_files(model_file_path, args_file_path, dream_team):
    model = torch.load(model_file_path,map_location='cpu')
    args = torch.load(args_file_path,map_location='cpu')
    model_name = args.model.upper()
    dataset_name = args.dataset.upper()
    layers = get_layers(model, dream_team=dream_team)
    no_layers = len(layers)
    return model, args, model_name, dataset_name, layers, no_layers

def write_message_to_file(msg, file_path, print_std=False):
    f = open(file_path, 'a')
    f.write(msg + "\n")
    f.close()
    if print_std:
        print(msg)

def np_nan(arr):
    arr[arr==0] = np.nan
    return arr

def reduce_layers(X, agg="mean"):
    if isinstance(agg, int):
        x = X[:, agg]
    elif agg == "eo_mean":
        x = X[:, range(0, X.shape[1], 2)].mean(1)
    elif agg == "mean":
        x = X.mean(1)
    elif agg == "median":
        x = np.median(X, 1)
    elif agg == "max":
        x = np.max(X, 1)
    else:
        raise ValueError
    return x

def get_dict_of_arrays(hist, ignore=[], ignore_terms=[]):
    # return {key: np.stack([h[key] for h in hist]) for key in hist[0].keys() if key not in ignore}
    r = {}
    for key in hist[0].keys():
        if key in ignore:
            continue
        if any([term in key for term in ignore_terms]):
            continue
        try:
            r[key] = np.stack([h[key] for h in hist])
        except:
            print(key)
            print([h[key] for h in hist])
            raise Exception
    return r

def npize(tensor):
    return tensor.cpu().detach().numpy()

def test_equality(x, y, atol):
    if atol == 0.0:
        return x == y
    else:
        return np.isclose(x, y, atol=atol)

def get_wrane_mask(dims, color_width, window_width):
    # dims = (width, height)
    mask = torch.zeros(dims, dtype=bool) # type: ignore
    outer_width = window_width - color_width
    mask[:, :window_width] = True
    mask[:, -window_width:] = True
    mask[:window_width, :] = True
    mask[-window_width:, :] = True
    outer_width = window_width - color_width
    if outer_width:
        mask[:, :outer_width] = False
        mask[:, -outer_width:] = False
        mask[:outer_width, :] = False
        mask[-outer_width:, :] = False
    return mask

import zlib
import hashlib
import random
## From ChatGPT
def rng_state_summary():
    # PyTorch RNG state
    pytorch_state = torch.get_rng_state().numpy()
    pytorch_checksum = zlib.crc32(pytorch_state)

    # Numpy RNG state
    numpy_state = np.random.get_state()[1]  # type: ignore # This is the state array
    numpy_checksum = zlib.crc32(numpy_state.view(np.uint8))

    # Python RNG state
    python_state = str(random.getstate()).encode('utf-8')  # Convert state to a string and then bytes
    python_checksum = int(hashlib.sha256(python_state).hexdigest(), 16) % (10 ** 8)  # Take SHA-256 hash and truncate for simplicity

    print(f"PyTorch RNG checksum: {pytorch_checksum}")
    print(f"Numpy RNG checksum: {numpy_checksum}")
    print(f"Python RNG checksum: {python_checksum}")

def get_logs(query="", num_exps=0, logs_path="logs/exp_register.log", cols=["action", "timestamp", "pid", "folder"], ignore_exp_ended_entries=True):
    df = pd.read_csv(logs_path, header=None)
    df.columns = cols
    df["exp_key"] = df.folder.str.split("/").str[-1]
    df["time"] = df.timestamp.astype(str).apply(lambda x: datetime.strptime(x[:-3], '%Y%m%d%H%M%S'))
    if ignore_exp_ended_entries:
        df = df.loc[df["action"] != "E"]
    if query:
        df = df.loc[df.folder.str.contains(query)]
    if num_exps:
        df = df.tail(num_exps)
    if query and num_exps:
        print(df.pid.values)
        [print(f'"{val}",') for val in df.exp_key.values]
    return df


def get_dual_norm(p):
    if p == 1:
        return np.inf
    return 1/(1-(1/p))



def is_linear_head(module, num_classes):
    return isinstance(module, nn.Linear) and (module.weight.shape[0] == num_classes)

import time

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self  # You can now access .elapsed or .interval
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
    @property
    def interval(self):
        return self.elapsed
    

class LRConv2d(nn.Module):
    """
    A Minimum Viable Product for a Low-Rank Convolutional Layer.

    This layer replaces a standard Conv2d with a sequence of two Conv2d layers:
    1. A spatial convolution that maps `in_channels` to an intermediate `rank` (r)
       using the original kernel size, stride, padding, dilation, and groups.
    2. A pointwise (1x1) convolution that maps from `rank` (r) to `out_channels`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size, rank: int,
                 stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = True,
                 conv1_bias: bool = False, padding_mode="zeros"): # Ensure padding_mode is an argument
        super(LRConv2d, self).__init__()

        if not isinstance(rank, int) or rank <= 0:
            raise ValueError("Rank 'rank' must be a positive integer.")

        if groups > 1:
            if in_channels % groups != 0:
                raise ValueError("in_channels must be divisible by groups")
            if rank % groups != 0:
                raise ValueError("Rank 'rank' must be divisible by 'groups' if groups > 1.")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv1_bias = conv1_bias
        self.padding_mode_param = padding_mode # Crucial: Store padding_mode

        # Convolution 1: Spatial convolution
        # Note: The internal nn.Conv2d will use its own padding_mode argument.
        # If you want LRConv2d's padding_mode to affect conv1, pass self.padding_mode_param here.
        # For now, it uses PyTorch's default 'zeros' unless padding_mode is explicitly passed to conv1.
        # The original nn.Conv2d's padding_mode is available in self.padding_mode_param.
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding, # This padding value is used
            dilation=dilation,
            groups=groups,
            bias=conv1_bias,
            padding_mode=self.padding_mode_param # Pass the stored padding_mode to the first conv layer
        )

        # Convolution 2: Pointwise (1x1) convolution
        self.conv2 = nn.Conv2d(
            in_channels=rank,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1, # Typically 1 for channel mixing, could be self.groups if rank and out_channels are divisible
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv1"):
            x = self.conv1(x)
            x = self.conv2(x)
        else:
            # HACK:
            x = self.lowrank_conv1(x)
            x = self.lowrank_conv2(x)
        return x

    def __repr__(self):
        s = (f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, "
             f"kernel_size={self.kernel_size}, rank={self.rank}, stride={self.stride}, "
             f"padding={self.padding}, dilation={self.dilation}, groups={self.groups}")
        if not self.bias:
            s += f", bias={self.bias}"
        if self.conv1_bias:
            s += f", conv1_bias={self.conv1_bias}"
        
        # Robustly check for padding_mode_param and if it's not the default
        if hasattr(self, 'padding_mode_param') and self.padding_mode_param != "zeros":
            s += f", padding_mode='{self.padding_mode_param}'"
        # If padding_mode_param is not set or is "zeros", it won't be added to the repr,
        # which is standard behavior for default parameters.
        
        s += ")"
        return s

def matricize_conv(layer, filter_matricization=True):
    if isinstance(layer, nn.Conv2d):
        weight = layer.weight
    elif isinstance(layer, LRConv2d):
        if hasattr(layer, "conv1"):
            weight = matricize_conv(layer.conv2.weight) @ matricize_conv(layer.conv1.weight)
        else:
            weight = matricize_conv(layer.lowrank_conv2.weight) @ matricize_conv(layer.lowrank_conv1.weight)
    elif isinstance(layer, torch.Tensor):
        weight = layer
    else:
        raise Exception
    fold_dim = 1 if filter_matricization else 2
    return torch.reshape(weight, (np.prod(weight.data.shape[:fold_dim]), np.prod(weight.data.shape[fold_dim:])))