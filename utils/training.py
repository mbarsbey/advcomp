import torch
import numpy as np
from .common import write_message_to_file, matricize_conv, is_linear_head
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from time import sleep
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.nn.utils import prune
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent
from transformers import get_wsd_schedule
from kron_torch import Kron


def write_experiment_script(script, file_traj, print_std):
    msg = "*** EXPERIMENT SCRIPT START ***\n" + script + "\n*** EXPERIMENT SCRIPT END ***\n"
    if script:
        write_message_to_file(msg, file_traj, print_std=print_std)

def get_accuracy_lower_bound(num_classes):
    return 1.25*(100/num_classes) if num_classes > 2 else 53

def update_avg_net(net, avg_net, num_iter, burn_in=1000):
    n = num_iter - burn_in + 1
    if num_iter < burn_in: # corrected this from <= to <
        return avg_net #changed this to use less memory #copy.deepcopy(net)
    else:
        with torch.no_grad():
            for (p_avg, p_new) in zip(avg_net.parameters(), net.parameters()):
                p_avg.data = (1 - 1 / n) * p_avg.data + (1 / n) * p_new.data
            return avg_net

def get_convergence_criteria(dataset, method, loss_crit, accuracy_crit):
    """Determining the convergence criteria according to determination method and/or dataset."""
    if method == "dataset":
        if dataset == "cifar100":
            loss_crit = 1e-2
            accuracy_crit = .99
        elif "dcase" in dataset:
            loss_crit = np.inf
            accuracy_crit = 95.
        elif "esc" in dataset:
            loss_crit = np.inf
            accuracy_crit = 100.
        else:
            loss_crit = 5e-5
            accuracy_crit = 100.
    if method == "none":
        loss_crit = -np.inf
        accuracy_crit = np.inf
    return loss_crit, accuracy_crit

def init_weights(m):
    if type(m) == nn.Linear:
        #nn.init.xavier_uniform_(m.weight)
        m.weight.data.fill_(0.01)
        #nn.init.uniform_(m.weight, a=-1e-6, b=1e-6)

def cycle_loader(dataloader):
    while 1:
        for data in dataloader:
            yield data

def get_optimizer(net, args):
    optim_args, param_names = [], []
    for i, (name, p) in enumerate(net.named_parameters()):
        optim_args.append({
            "params": p,
            "lr": args.lr[i]  if len(args.lr) > 1 else args.lr[0],
            "momentum": args.momentum,
            "weight_decay": args.wd[i] if len(args.wd) > 1 else args.wd[0],
            "dampening": getattr(args, "dampening", 0.0)
            })
        param_names.append(name)

    if getattr(args, "llwd", 0.0) > 0.0:
        assert args.lr_algorithm == "sgd"
        assert args.use_bias == False
        if "resnet" in args.model:
            for i, name in enumerate(param_names):
                if name == "fc.weight":
                    optim_args[i]["weight_decay"] = args.llwd
                    break
        elif ("vgg" in args.model) or ("fcn" in args.model):
            optim_args[-1]["weight_decay"] = args.llwd
        else:
            raise NotImplementedError

    if args.lr_algorithm == "sgd":
        opt = optim.SGD(optim_args)
    elif args.lr_algorithm == "adam":
        for optim_arg in optim_args:
            del optim_arg["momentum"]
            optim_arg["betas"] = args.adam_b1, args.adam_b2
        opt = optim.AdamW(optim_args)
    elif args.lr_algorithm == "kron":
        for optim_arg in optim_args:
            del optim_arg["momentum"]
        opt = Kron(optim_args)
    else:
        raise NotImplementedError
    if "sam" in getattr(args, "optimizer", "sgd"):
        assert args.lr_algorithm == "sgd"
        rho = float(args.optimizer.split("-")[1]) if "-" in args.optimizer else 2.0
        adaptive = "asam" in args.optimizer
        opt = SAM(net.parameters(), optim.SGD, rho=rho, adaptive=adaptive, lr=args.lr[0], momentum=args.momentum, weight_decay=args.wd[0]) # HACK WD
        # scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    return opt

def get_criterion(args, reduction=None):
    if reduction is None:
        reduction = "mean"
    if args.criterion.upper() == 'NLL':
        crit = nn.CrossEntropyLoss(reduction=reduction).to(args.device)
    elif args.criterion.upper() == 'MSE':
        crit = nn.MSELoss(reduction=reduction)
    elif args.criterion.upper() == 'BCE':
        crit = nn.BCEWithLogitsLoss(reduction=reduction)
    elif 'floss' in args.criterion:
        _, gamma = args.criterion.split("-")
        crit = FocalLoss(gamma=float(gamma), reduction=reduction)
    else:
        raise KeyError
    return crit

def get_scheduler(scheduler, optimizer, milestones, gamma):
    if scheduler == "multisteplr":
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif "cosine" in scheduler:
        _, max_epochs, eta_min = scheduler.split("-")
        max_epochs, eta_min = int(max_epochs), float(eta_min)
        return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)
    elif "wsd" in scheduler:
        _, num_warmup_steps, num_stable_steps, num_decay_steps = scheduler.split("-")
        num_warmup_steps, num_stable_steps, num_decay_steps = float(num_warmup_steps), float(num_stable_steps), float(num_decay_steps)
        return get_wsd_schedule(optimizer, num_warmup_steps=num_warmup_steps, num_stable_steps=num_stable_steps, num_decay_steps=num_decay_steps)
    else:
        raise KeyError


class ReLUSigmoid(nn.Module):
    def __init__(self, shift=-0.5, scale=2.0):
        super(ReLUSigmoid, self).__init__()
        # Initialize shift and scale as fixed values
        self.shift = shift
        self.scale = scale

    def forward(self, x):
        return (torch.sigmoid(x = F.relu(x)) + self.shift) * self.scale # type: ignore


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "lerelu":
        return nn.LeakyReLU()
    elif activation == "lerelu01":
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == "tanh":
        return nn.Tanh()      
    elif activation == "linear":
        return nn.Identity()  
    elif activation == "relusigmoid":
        return ReLUSigmoid()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise KeyError("Activation function not recognized.")


#from ChatGPT
def set_track_running_stats(model, value=False):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = value

def generate_x_adv(net, x_orig, p, input_shape, num_classes,  stats, crit, device, attack_eps=0.0314, attack_lr=0.05882, attacks_max_iter=10, seed=0, attacker="pgd", evaluation=False):
    x_mean, x_std = np.array(stats["mean"])[np.newaxis, :, np.newaxis, np.newaxis], np.array(stats["std"])[np.newaxis, :, np.newaxis, np.newaxis]
    x = (x_orig * x_std) + x_mean
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net.eval()
        
        net_cls = PyTorchClassifier(
            net,
            loss=crit, 
            input_shape=input_shape, 
            nb_classes=num_classes, 
            optimizer=None,
            preprocessing=(stats["mean"], stats["std"]) # type: ignore
            )
        if attacker == "pgd":                    
            attacks = ProjectedGradientDescent(net_cls, norm={"2": 2, "inf": np.inf}[p], eps=attack_eps, eps_step=attack_lr, max_iter=30 if evaluation else attacks_max_iter, num_random_init=10 if evaluation else 1, verbose=False)
        else:
            raise ValueError
        x_adv = attacks.generate(x=x.astype(np.float32)) # Pass NCHW
        x_adv = (x_adv - x_mean) / x_std
    return torch.tensor(x_adv).float().to(device), torch.tensor(x_orig).float().to(device)