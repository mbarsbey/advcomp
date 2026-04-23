# On the Interaction of Compressibility and Adversarial Robustness

Source code for the paper *"On the Interaction of Compressibility and Adversarial Robustness"*, presented at the International Conference on Learning Representations (ICLR) 2026.

## Overview

This repository provides a training and evaluation pipeline for studying how various notions of compressibility - structured prunability, low effective rank - interact with adversarial robustness. The pipeline supports a range of architectures, regularizers, and adversarial training settings, and tracks compressibility statistics over the course of training.

## Installation

```bash
conda create -n advcomp python=3.10
conda activate advcomp
pip install -r requirements.txt
```

The `requirements.txt` is a full conda environment dump; you may not need every package depending on which experiments you run. Core dependencies include PyTorch, torchvision, JAX, captum, adversarial-robustness-toolbox, and pyhessian.

## Quick Start

A minimal example is provided in `experiment.sh`:

```bash
bash experiment.sh
```

This trains a 5-layer FCN of width 2000 on CIFAR-10 with Adam, weight decay, and validation-based model selection.

## Key Arguments

`main.py` exposes a large set of options. The most relevant groups:

**Model & data**
- `--model` - `fcn`, or any torchvision classifier (e.g., `resnet18-tv`, `vgg16-tv`); append `-pretrained` to load ImageNet weights.
- `--width`, `--depth` - FCN dimensions.
- `--dataset` - `mnist`, `cifar10`, `cifar100`, `svhn`.
- `--data_augmentation` - `none` or `full` (RandomCrop + Flip + ColorJitter + Affine).

**Optimization**
- `--lr_algorithm` - `sgd`, `adam`, `kron`.
- `--lr`, `--wd`, `--momentum`.
- `--lr_scheduler` - e.g., `multisteplr`, `cosine-<T_max>-<eta_min>`.
- `--iterations`, `--batch_size_train`.

**Regularization toward compressibility**
- `--l1 <lambda>` - L1 penalty on parameters.
- `--rowwise_l2 <lambda>` (a.k.a. `--group_lasso`) - row-wise L2 (group sparsity).
- `--nuclear_norm <lambda>` - nuclear-norm penalty.
- `--layer_rank <r>` - replace dense / conv layers with explicit rank-`r` factorizations.

**Adversarial training**
- `--at_norm` - `L2` or `inf`.
- `--at_eps`, `--at_lr`, `--at_iters` - PGD parameters.
- `--at_ratio` - fraction of each batch perturbed.
- `--at_attacker` - currently `pgd`.

**Validation-based selection**
- `--val_split val --val_criterion loss --val_patience 5000 --val_keep_best_model`

**Compressibility tracking**

Pass any number of statistics via `--track_statistics`. Examples:
- `par_comp_0.1` - parameter compressibility at sparsity level 0.1.
- `net_comp_0.1` - network-wide compressibility.
- `par_er` - per-layer effective rank.
- `act_er_train_m1_1000` - activation effective rank at layer −1, 1000 train samples.
- `act_phdim_train_m2_1000_128` - persistent-homology dimension of activations.

## Example Configurations

**FCN on CIFAR-10 with rank constraint:**
```bash
python main.py --model fcn --dataset cifar10 --width 2000 --depth 5 \
  --layer_rank 15 --lr_algorithm adam --lr 0.001 --wd 0.01 \
  --val_split val --val_criterion loss --val_patience 5000 --val_keep_best_model
```

**ResNet-18 with PGD adversarial training (L∞):**
```bash
python main.py --model resnet18-tv --dataset cifar10 --aug full \
  --at_norm inf --at_eps 0.0314 --at_lr 0.00784 --at_iters 10 --at_ratio 0.5 \
  --lr_algorithm sgd --lr 0.1 --wd 5e-4 --momentum 0.9 \
  --lr_scheduler multisteplr --lrs_m 100 150 --lrs_g 0.1
```

## Outputs

Each run writes to `--save_dir`:
- `args.info`, `script.sh` - run configuration.
- `training_history.hist`, `evaluation_history.hist` - per-iteration logs (loss, accuracy, gradient and parameter norms, tracked statistics).
- `net.pyT` - final model.
- `val_iters/` - top-k validation checkpoints (when validation selection is enabled).

## Coverage

The code in this repository replicates the main findings of the paper. If there is a specific experiment you would like to reproduce but cannot find here, please contact Melih Barsbey at `m.barsbey@imperial.ac.uk`.

## Citation

```
@inproceedings{barsbeyInteractionCompressibility2026,
  title     = {On the Interaction of Compressibility and Adversarial Robustness},
  author    = {Barsbey, Melih and Ribeiro, Ant{\^o}nio H. and Simsekli, Umut and Birdal, Tolga},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
