import torch
import torch.nn as nn
import numpy as np
from utils import get_activation
from torchvision.models import ResNet
from resnet18 import LRConv2d
from functools import partial


def replace_conv_with_lr(module: nn.Module, rank: int) -> nn.Module:
    """
    Recursively traverses a PyTorch nn.Module and replaces all nn.Conv2d layers
    with LRConv2d layers using the specified rank.

    Args:
        module (nn.Module): The PyTorch model (or a submodule) to modify.
        rank (int): The rank to use for the LRConv2d layers. This rank must be
                    a positive integer. If the original Conv2d layer uses groups > 1,
                    'rank' must be divisible by 'groups'.

    Returns:
        nn.Module: The modified module with nn.Conv2d layers replaced.
                   The modification is done in-place.
    """
    for name, child_layer in module.named_children():
        
        if isinstance(child_layer, nn.Conv2d):
            in_channels = child_layer.in_channels
            if in_channels <= 3:
                continue
            out_channels = child_layer.out_channels
            kernel_size = child_layer.kernel_size
            stride = child_layer.stride
            padding = child_layer.padding
            dilation = child_layer.dilation
            groups = child_layer.groups
            original_bias_enabled = child_layer.bias is not None
            padding_mode = child_layer.padding_mode # Get padding_mode from original Conv2d

            new_lr_layer = LRConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                rank=rank,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=original_bias_enabled,
                # conv1_bias uses its default from LRConv2d.__init__ (False)
                padding_mode=padding_mode # Pass it to LRConv2d constructor
            )
            
            setattr(module, name, new_lr_layer)
        else:
            replace_conv_with_lr(child_layer, rank)
            
    return module


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=False, equiv_fro_norm_init=False):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(out_features, rank))
        self.B = nn.Parameter(torch.randn(rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters(equiv_fro_norm_init)

    def reset_parameters(self, equiv_fro_norm_init):
        if equiv_fro_norm_init:
            out_features, r = self.A.shape
            r_check, in_features = self.B.shape
            assert r == r_check
            scale = 0.5 * (out_features / (r * (in_features + out_features))) ** 0.25
            nn.init.normal_(self.A, mean=0.0, std=scale)
            nn.init.normal_(self.B, mean=0.0, std=scale)
            print(self.A)
        else:
            nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
            nn.init.kaiming_uniform_(self.B, a=np.sqrt(5))
        if self.bias is not None:
            fan_in = self.B.size(1)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight = self.A @ self.B
        if self.bias is not None:
            return torch.nn.functional.linear(x, weight, self.bias)
        else:
            return torch.nn.functional.linear(x, weight)
    
    @property
    def weight(self):
        return self.A @ self.B

class MultiLayerNN(nn.Module):

    def __init__(self, input_dim=28*28, width=50, depth=2, num_classes=10, activation="relu", bias=False, get_learned_repr=False, top_k_activations=0, top_k_neurons=0, ablate_top_k_neurons=0, activation_exponent=0.0, dropout=0.0, **kwargs):
        assert depth >= 1 
        super(MultiLayerNN, self).__init__()
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.get_learned_repr = get_learned_repr
        self.top_k_activations = top_k_activations
        self.top_k_neurons = top_k_neurons
        self.ablate_top_k_neurons = ablate_top_k_neurons
        self.activation_exponent = activation_exponent
        self.dropout = dropout
        
        Linear = partial(LowRankLinear, rank=kwargs["layer_rank"], bias=bias) if kwargs["layer_rank"] > -1 else partial(nn.Linear, bias=bias)

        num_output_dims = 1 if kwargs.get("single_logit", False) else self.num_classes
        layers = []
        for i in range(depth-1):
            layers.append(Linear(self.width if i > 0 else self.input_dim, self.width))
            layers.append(get_activation(activation))
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout)) 

        if getattr(self, "depth", 2) > 1:
            layers.append(nn.Linear(self.width, num_output_dims, bias=bias))
        elif getattr(self, "depth", 2) == 1:
            layers.append(nn.Linear(self.input_dim, num_output_dims, bias=bias))
        else:
            raise ValueError
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # x = x.view(x.size(0), self.input_dim)
        x = x.reshape(x.size(0), self.input_dim)
        if getattr(self, "depth", 2) > 1:
            x = self.fc[:-1](x)

        x = self.fc[-1](x)
        return x

def get_model(model, device, input_dim, width, depth, num_classes, activation, bias, llnobias=False, img_channels=3, dropout=0.0, get_all_repr_norms=0, **kwargs):
    if "fcn" in model:
        net = MultiLayerNN(input_dim=input_dim, width=width, depth=depth, num_classes=num_classes, activation=activation, bias=bias, dropout=dropout, **kwargs).to(device)

        if kwargs.get("custom_init", False):
            #net.apply(init_weights)
            with torch.random.fork_rng():
                # Set a local seed
                torch.manual_seed(0)
                p_init = list(torch.load(kwargs["custom_init"]).parameters())[-1]
                p_shape, p_numel = p_init.shape, p_init.numel()
                p = list(net.parameters())[-1]
                p.data = torch.reshape(p_init.flatten()[torch.randperm(p_numel)], p_shape)
        if kwargs.get("custom_init", False):
            for i, param in enumerate(net.parameters()):
                # HACK
                if num_classes in param.shape:
                    param.requires_grad = False
                    param.data.fill_(1/width)
    else:
        from torchvision import models as tm

        net = getattr(tm, model.replace("-tv", "").replace("-pretrained", ""))(weights="DEFAULT" if "pretrained" in model else None)
        

        # Making sure fc is the last layer in the given architecture, by testing ImageNet-1k output
        assert net.fc.out_features == 1000
        net.fc = nn.Linear(net.fc.in_features, num_classes, bias=net.fc.bias is not None)
        if img_channels != 3:
            net.conv1 = nn.Conv2d(img_channels, net.conv1.out_channels, kernel_size=net.conv1.kernel_size, padding=net.conv1.padding, bias=net.conv1.bias is not None)
        if isinstance(net, ResNet):
            if dropout > 1e-6:
                net = ModifiedResNet(net, -1, dropout=dropout)
            if llnobias:
                net.fc.bias = None
            if kwargs["layer_rank"] > -1:
                # No linear layers in torchvision ResNet models other than the classifier head
                replace_conv_with_lr(net, kwargs["layer_rank"])
                
                
        net = net.to(device)
    return net
def get_grad_norms(net):
    return [p.grad.detach().data.norm() for p in net.parameters()]
