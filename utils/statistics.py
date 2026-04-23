import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax.scipy.special import entr
import jax
import jax.numpy as jnp
from functools import partial
import shutil
import torch
from .common import get_layers, test_equality, get_wrane_mask, get_timestamp
from .data import get_data
import numpy as np
from captum.attr import DeepLift, IntegratedGradients, Saliency, InputXGradient
from captum.attr import visualization as viz
from copy import deepcopy
from pyhessian import hessian # Hessian computation
from itertools import islice
from sklearn.decomposition import PCA
import pandas as pd
from captum.attr import LayerConductance
from resnet18 import ResNet18, ModifiedResNet
from torchvision import datasets
from torchvision.models import ResNet, VGG, SwinTransformer
import torch.nn as nn


@jax.jit
def get_effective_rank(z):
  if len(z.shape) == 4:
    z = z.reshape(np.prod(z.shape[:2]), np.prod(z.shape[2:]))
  s = jnp.linalg.svd(z, compute_uv=False)
  s = s/s.sum()
  return entr(s).sum()

@jax.jit
def get_stable_rank(X):
  return (jnp.linalg.norm(X, ord="fro") ** 2)/(jnp.linalg.norm(X, ord=2) ** 2)

get_idx = lambda kappa, d: int((1-kappa) * d)

def attribute_image_features(algorithm, net, input, target, baselines):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input, target=target, baselines=baselines)
    return tensor_attributions

def get_pqi(z, p, q, d):
    pqi = 1 - (d ** (1/q - 1/p)) * (jnp.linalg.norm(z, ord=p)/jnp.linalg.norm(z, ord=q))
    pqi = jnp.where(pqi == -float('inf'), 0, pqi)
    pqi = jnp.where(jnp.logical_and(pqi > - 1e-5, pqi < 0), 0, pqi)
    return pqi
get_pqi = jax.jit(get_pqi, static_argnums=[1,2,3])
get_pqi = jax.vmap(get_pqi, in_axes=[0, None, None, None])

# def rel_best_k_appx_error(ŵ, q, idx):
#     argsorted_ŵ = jnp.argsort(jnp.abs(ŵ))
#     ŵ_appx = ŵ.at[argsorted_ŵ[:idx]].set(0.)
#     return jnp.linalg.norm(ŵ_appx - ŵ, q) / jnp.linalg.norm(ŵ, q)
# rel_best_k_appx_error = jax.jit(rel_best_k_appx_error, static_argnums=[1,2])
# rel_best_k_appx_error = jax.vmap(rel_best_k_appx_error, in_axes=[0, None, None])

def get_iters(iters_folder):
    iters = [int(it.replace(".pyT", "").split("_")[-1]) for it in os.listdir(iters_folder)] + [-1]
    iters = sorted(iters)
    if iters[0] == -1:
      iters = iters[1:] + [-1] 
    return iters

# def get_compressibility(layer, kappa, q, flat):
#     if flat:
#       layer = layer.flatten()[np.newaxis, :]
#     elif layer.ndim == 4:
#       layer = layer.reshape(np.prod(layer.shape[:2]), np.prod(layer.shape[2:]))
#     return 1-rel_best_k_appx_error(layer, q, get_idx(kappa, layer.shape[1])).mean()

def rel_best_k_appx_error_orig(ŵ, q, idx):
    _, top_indices = jax.lax.top_k(jnp.abs(ŵ), idx)
    mask = jnp.ones_like(ŵ, dtype=bool).at[top_indices].set(False)
    return jnp.linalg.norm(ŵ * mask, q) / jnp.linalg.norm(ŵ, q)
rel_best_k_appx_error_v = jax.jit(rel_best_k_appx_error_orig, static_argnums=[1,2])
rel_best_k_appx_error = jax.vmap(rel_best_k_appx_error_v, in_axes=[0, None, None])


def get_vector_compressibility(ŵ, q, kappa):
  return 1-rel_best_k_appx_error_v(ŵ, q, len(ŵ) - get_idx(kappa, len(ŵ) + 1))

def prune_vector(ŵ, kappa):
  idx = len(ŵ) - get_idx(kappa, len(ŵ))
  _, top_indices = jax.lax.top_k(jnp.abs(ŵ), idx)
  mask = jnp.zeros_like(ŵ, dtype=bool).at[top_indices].set(1)
  return ŵ * mask                           

def get_compressibility(layer, kappa, q, flat, weighted=False):
    if flat:
      layer = layer.flatten()[np.newaxis, :]
    elif layer.ndim == 1:
      layer = layer[np.newaxis, :]
    elif layer.ndim == 4:
      layer = layer.reshape(np.prod(layer.shape[:1]), np.prod(layer.shape[1:]))
    if not weighted:
      return 1-rel_best_k_appx_error(layer, q, layer.shape[1] - get_idx(kappa, layer.shape[1])).mean()
    else:
      comps = rel_best_k_appx_error(layer, q, layer.shape[1] - get_idx(kappa, layer.shape[1]))
      norms = jnp.linalg.norm(layer, ord=q, axis=1)
      norms /= norms.sum()
      return 1 - jnp.dot(comps, norms)

def get_network_compressibility(layers, kappa, q):
    all_pars = np.concatenate([layer.flatten() for layer in layers])[np.newaxis, :]
    return 1-rel_best_k_appx_error(all_pars, q, all_pars.shape[1] - get_idx(kappa, all_pars.shape[1])).mean()

def get_ph_dim(layer, k):
   if layer.ndim == 1:
      return np.nan
   if layer.ndim == 4:
      layer = layer.reshape(np.prod(layer.shape[:1]), np.prod(layer.shape[1:]))
   if layer.shape[0] < layer.shape[1]:
      layer = layer.T
   if k:
      layer_ld = PCA(n_components=min(k, layer.shape[1])).fit_transform(layer)
   return topology.calculate_ph_dim(layer_ld) 

def is_bias_dataset(dataset):
  if isinstance(dataset, datasets.CIFAR10) or isinstance(dataset, datasets.CIFAR100) or isinstance(dataset, datasets.MNIST) or isinstance(dataset, datasets.ImageFolder):
    return False
  else:
    return True  


def get_samples_from_loader(data_loader, num_samples, seed=0, device=None, bias=False, get_idx=False):
    batch_size = data_loader.batch_size 
    num_batches = num_samples // batch_size
    if (num_samples % batch_size):
        num_batches += 1
        
    with torch.random.fork_rng():
        # Set a local seed
        torch.manual_seed(seed)
        data_loader_iter = iter(data_loader)
        samples = list(islice(data_loader_iter, num_batches))
    samples = samples[:num_samples]
    idx, images, labels = torch.concatenate([sample[0] for sample in samples]), torch.vstack([sample[1] for sample in samples]), torch.concatenate([sample[2] for sample in samples])
    result = images, labels
    if get_idx:
        result = idx, *result
    if bias:
      # dataset.dataset for cifar10 and cifar100
      if is_bias_dataset(getattr(data_loader.dataset, "dataset", None)):
        bias = torch.concatenate([sample[3] for sample in samples])
      else:
        # HACK
        bias = labels
      result = *result, bias
    if device:
      result = (res.to(device) for res in result)
    return result

def get_attribution_function(attribution_method):
    if (attribution_method == "deeplift") or (attribution_method == "dl"):
        return DeepLift
    elif (attribution_method == "integrated_gradients") or (attribution_method == "ig"):
        return IntegratedGradients 
    elif (attribution_method == "saliency") or (attribution_method == "s"):
        return Saliency 
    elif (attribution_method == "input_x_gradient") or (attribution_method == "ixg"):
        return InputXGradient
    else:
       raise KeyError("Attribution method cannot be found.")

def get_stable_rank(X):
  return (jnp.linalg.norm(X, ord="fro") ** 2)/(jnp.linalg.norm(X, ord=2) ** 2)

def get_layer_conductance(net, images, labels, num_classes, layer_idx, kappa, return_all=False):
  if hasattr(net, "features"):
     lc = LayerConductance(net, net.features.module[layer_idx])
  else:
     lc = LayerConductance(net, net.fc[layer_idx])
  attr = lc.attribute(images, target=labels, internal_batch_size=1) #
  if len(attr.shape) > 2:
     attr = torch.reshape(attr, (attr.shape[0], -1))
  class_attrs = np.zeros((num_classes, attr.shape[1]))
  for i in range(num_classes):
      class_attrs[i, :] = torch.abs(attr[labels==i]).mean(0).cpu().detach().numpy()
  c = class_attrs.T
  num_units = int(c.shape[0] * kappa)
  mag_idx = np.flip(np.argsort(class_attrs.sum(0)))
  most_class_attrs = c[mag_idx[:num_units]]
  most_class_attrs /= most_class_attrs.sum(1, keepdims=True) + 1e-8
  if return_all:
    return most_class_attrs, c
  return most_class_attrs

def get_cosine_dist(z, v=None):
    z_normalized = z / z.norm(p=2, dim=1, keepdim=True)
    if v is None:
        return 1 - torch.mm(z_normalized, z_normalized.t()).mean()
    else:
        v_normalized = v / v.norm(p=2, dim=1, keepdim=True)
        return 1 - torch.mm(z_normalized, v_normalized.t()).mean()

def get_manhattan_dist(z, v=None):
    z_bin = (z > 0).float() 
    if v is None:
        return torch.abs(z_bin.unsqueeze(1) - z_bin.unsqueeze(0)).sum(dim=2).mean()/z.shape[1]
    else:
        v_bin = (v > 0).float() 
        return torch.abs(z_bin.unsqueeze(1) - v_bin.unsqueeze(0)).sum(dim=2).mean()/z.shape[1]

def get_l1_dist(z, v=None):
    z_normalized = z / z.norm(p=1, dim=1, keepdim=True)
    z_normalized = z
    if v is None:
        return torch.abs(z_normalized.unsqueeze(1) - z_normalized.unsqueeze(0)).sum(dim=2).mean()
    else:
        v_normalized = v / v.norm(p=1, dim=1, keepdim=True)
        v_normalized = v
        return torch.abs(z_normalized.unsqueeze(1) - v_normalized.unsqueeze(0)).sum(dim=2).mean()

def get_class_separation(z, y, num_classes, similarity="cosine", median=False, return_dist_matrix=False):
    if similarity == "cosine":
      get_dist = get_cosine_dist
    elif similarity == "manhattan":
      get_dist = get_manhattan_dist
    elif similarity == "l1":
      get_dist = get_l1_dist
    else:
       raise KeyError
    d_within_tensor = torch.tensor([get_dist(z[y == i]) for i in range(num_classes)])
    d_between_tensor = torch.tensor([get_dist(z[y == i], z[y==j]) for i in range(num_classes) for j in range(num_classes)])
    if median:
      d_within = torch.nanmedian(d_within_tensor)
      d_between = torch.nanmedian(d_between_tensor)
    else:
      d_within = torch.nanmean(d_within_tensor)
      d_between = torch.nanmean(d_between_tensor)
    if return_dist_matrix:
      return (1-d_within/d_between).item(), d_within.item(), d_between.item(), torch.reshape(d_between_tensor, (num_classes, num_classes)).detach().cpu().numpy()
    return (1-d_within/d_between).item(), d_within.item(), d_between.item()

def normalize_representations(z, within=None, p=2):
    if within is None:
      return z
    if within == "samples":
      return z / z.norm(p=p, dim=1, keepdim=True)
    if within == "neurons":
      return z / z.norm(p=p, dim=0, keepdim=True)

def get_l1_dist_per_neuron(z, v=None):
    if v is None:
        return torch.abs(z.unsqueeze(1) - z.unsqueeze(0)).mean(dim=(0, 1))
    else:
        return torch.abs(z.unsqueeze(1) - v.unsqueeze(0)).mean(dim=(0, 1))

def get_sum_l1_dist_per_neuron(z, v=None):
    if v is None:
        return torch.abs(z.unsqueeze(1) - z.unsqueeze(0)).sum(dim=(0, 1))
    else:
        return torch.abs(z.unsqueeze(1) - v.unsqueeze(0)).mean(dim=(0, 1))

def get_l2_dist_per_neuron(z, v=None):
    if v is None:
        return torch.sqrt((torch.abs(z.unsqueeze(1) - z.unsqueeze(0))**2).mean(dim=(0, 1)))
    else:
        return torch.sqrt((torch.abs(z.unsqueeze(1) - v.unsqueeze(0))**2).mean(dim=(0, 1)))

def get_class_separation_per_neuron(z, y, num_classes, similarity="l1", maximum=False, normalize_within=None):
    if similarity == "l1":
      get_dist = get_l1_dist_per_neuron
      z = normalize_representations(z, within=normalize_within, p=1)
    elif similarity == "sum_l1":
      get_dist = get_sum_l1_dist_per_neuron
      z = normalize_representations(z, within=normalize_within, p=1)
    elif similarity == "l2":
      get_dist = get_l1_dist_per_neuron
      z = normalize_representations(z, within=normalize_within, p=2)
    else:
      raise KeyError
    if maximum:
      tensor_within = torch.stack([get_dist(z[y == i]) for i in range(num_classes)]) #.max(0).values
      tensor_between = torch.reshape(torch.stack([get_dist(z[y == i], z[y==j]) for i in range(num_classes) for j in range(num_classes)]), (num_classes, num_classes, -1)).mean(0)
      return 1 - (tensor_within / tensor_between).min(0).values
      # matrix_d_between = torch.stack([get_dist(z[y == i], z[y==j]) for i in range(num_classes) for j in range(num_classes)])
      # d_between = torch.sort(matrix_d_between).values[-10:].mean(0)
    else:
      d_within = torch.stack([get_dist(z[y == i]) for i in range(num_classes)]).mean(0)
      d_between = torch.stack([get_dist(z[y == i], z[y==j]) for i in range(num_classes) for j in range(num_classes)]).mean(0)
    return torch.nan_to_num(1-d_within/d_between)

def get_output(model, images, labels, batch_size, device, spurs=None, eval_mode=True, criterion=None):
    if criterion == "nll":
      crit = nn.CrossEntropyLoss(reduction='none').to(device)
    elif criterion is not None:
      raise ValueError
    all_y_hat_list = []
    all_out_list = []
    if eval_mode:
      model.eval()
    # go through all the batches in the dataset
    num_iterations = len(images) // batch_size if not len(images) % batch_size else len(images) // batch_size + 1
    for b in range(num_iterations):
        X, y = images[b * batch_size:(b+1) * batch_size], labels[b * batch_size:(b+1) * batch_size]
        if not eval_mode:
          model.train()
        with torch.no_grad():
          out = model(X)
        if not eval_mode:
          model.eval()
        _, y_hat = out.max(1)
        all_out_list.append(out)
        all_y_hat_list.append(y_hat.flatten())
    y_hat = torch.hstack(all_y_hat_list)
    logits = torch.vstack(all_out_list)
    if criterion == "nll":
      with torch.no_grad():
        losses = crit(logits, labels)
    else:
        losses = np.nan
    return y_hat, logits, losses

def get_representations(model, layer_idx, images, labels, batch_size, spurs=None, logits=False, eval_mode=False):
    if eval_mode:
      model.eval()
    else:
      model.train()
    activation = {}
    def getActivation(name):
    # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    if isinstance(model, CustomVGG):
      # HACK
      if hasattr(model, "avgpool"):
        if layer_idx < 0:
          layer_idx += 1
          if layer_idx == 0:
            h1 = model.avgpool.register_forward_hook(getActivation('desired_relu'))   
          else:
            h1 = model.features.module[layer_idx].register_forward_hook(getActivation('desired_relu'))
      else:
        h1 = model.features.module[layer_idx].register_forward_hook(getActivation('desired_relu'))
    # register forward hooks on the layers of choice
    elif isinstance(model, VGG):
      if layer_idx == -1:
        h1 = model.classifier[4].register_forward_hook(getActivation('desired_relu')) 
    elif isinstance(model, SwinTransformer):
      if layer_idx == -1:
        h1 = model.flatten.register_forward_hook(getActivation('desired_relu')) 
    elif isinstance(model, ResNet18) or isinstance(model, ResNet):
      if layer_idx == -1:
        h1 = model.avgpool.register_forward_hook(getActivation('desired_relu'))
      elif layer_idx == 0:
        h1 = model.relu.register_forward_hook(getActivation('desired_relu'))
      elif layer_idx == -2:
        h1 = model.layer4[1].bn2.register_forward_hook(getActivation('desired_relu'))
      elif layer_idx == -3:

        h1 = model.layer4[1].relu.register_forward_hook(getActivation('desired_relu'))
      elif layer_idx == -4:
        h1 = model.layer3[1].register_forward_hook(getActivation('desired_relu'))
      else:
         raise NotImplementedError
    elif isinstance(model, ModifiedResNet):
      if layer_idx == -1:
        h1 = model.model.avgpool.register_forward_hook(getActivation('desired_relu'))
    else:
      h1 = model.fc[layer_idx].register_forward_hook(getActivation('desired_relu'))

    last_relu_list = []
    all_s_list = []
    all_y_list = []
    all_y_hat_list = []
    all_out_list = []
    # go through all the batches in the dataset
    num_iterations = len(images) // batch_size if not len(images) % batch_size else len(images) // batch_size + 1
    for b in range(num_iterations):
        X, y = images[b * batch_size:(b+1) * batch_size], labels[b * batch_size:(b+1) * batch_size]
        # forward pass -- getting the outputs
        # TODO: Fix obtaining the representations with the model in train mode.
        # with torch.no_grad:
        out = model(X)
        _, y_hat = out.max(1)
        if logits:
          all_out_list.append(out)
        # collect the activations in the correct list
        last_relu_list.append(activation['desired_relu'])
        all_y_list.append(y)
        all_y_hat_list.append(y_hat.flatten())
        if spurs is not None:
          all_s_list.append(spurs[b * batch_size:(b+1) * batch_size])
    # detach the hooks
    h1.remove()
    del h1
    z = torch.vstack(last_relu_list)
    z = z.cpu().numpy()
    y = torch.hstack(all_y_list).cpu().numpy()
    y_hat = torch.hstack(all_y_hat_list).cpu().numpy()
    if logits:
      out = torch.vstack(all_out_list)
    del model
    if spurs is not None: 
      s = torch.hstack(all_s_list).cpu().numpy()
      return z, y, y_hat, s
    if (spurs is not None) and logits: 
      s = torch.hstack(all_s_list).cpu().numpy()
      return z, y, y_hat, s, out
    if logits:
      return z, y, y_hat, out
    return z, y, y_hat

def compute_csi_bsi_neurons(z, y, s, num_classes):
    class_mean_activations = np.stack([z[y==i].mean(0) for i in range(num_classes)]).T
    bias_mean_activations = np.stack([z[s==i].mean(0) for i in range(num_classes)]).T    
    class_ratio_activations = np.stack([(z[y==i] > 0).mean(0) for i in range(num_classes)]).T
    bias_ratio_activations = np.stack([(z[s==i] > 0).mean(0) for i in range(num_classes)]).T
    group_mean_activations = np.stack([z[(y==i) & (s==j)].mean(0) for i in range(num_classes) for j in range(num_classes)]).T
    group_mean_activations = np.reshape(group_mean_activations, (z.shape[1], num_classes, num_classes))
    csi_within_bias = np.stack([np.apply_along_axis(compute_class_selectivity_index, 1, group_mean_activations[:,:,i]) for i in range(num_classes)]).T
    csi, max_idx_csi = np.apply_along_axis(partial(compute_class_selectivity_index, return_max_idx=True), 1, class_mean_activations).T
    weighted_csi = csi * class_ratio_activations[np.arange(len(max_idx_csi)), max_idx_csi.astype(int)]
    bsi, max_idx_bsi = np.apply_along_axis(partial(compute_class_selectivity_index, return_max_idx=True), 1, bias_mean_activations).T
    weighted_bsi = bsi * bias_ratio_activations[np.arange(len(max_idx_bsi)), max_idx_bsi.astype(int)]
    # max_csi_in_bias = weighted_bsi * (1-csi_within_bias)[np.arange(len(csi_within_bias)), max_idx_bsi.astype(int)]
    csi_in_bias = ((bias_mean_activations/bias_mean_activations.sum(1, keepdims=True)) * (csi_within_bias)).sum(1)
    csi_in_bias[np.isnan(csi_in_bias)] = 0.0
    return weighted_csi, weighted_bsi, csi_in_bias, csi, bsi

def compute_avg_csi_in_bias(z, y, s, num_classes, active_neurons_threshold=0.0, bias_quantile_threshold=0.9):
    weighted_csi, weighted_bsi, csi_in_bias, csi, bsi = compute_csi_bsi_neurons(z, y, s, num_classes)
    bsi_above_threshold = weighted_bsi > np.quantile(weighted_bsi, q=bias_quantile_threshold)
    z_s_neurons = (z!=0).sum(0)/z.shape[0]
    act_neur = (z_s_neurons > active_neurons_threshold)
    csi_gt_bsi = (weighted_csi > weighted_bsi)
    csi_gt_bsi_ratio = csi_gt_bsi[act_neur].mean()
    avg_csi_in_bias = csi_in_bias[act_neur & ~csi_gt_bsi].mean()
    avg_csi_in_tobias = csi_in_bias[act_neur & bsi_above_threshold].mean()
    return csi_gt_bsi_ratio, avg_csi_in_bias, act_neur.mean(), (z>0.0).mean(), avg_csi_in_tobias, weighted_csi[act_neur].mean(), weighted_bsi[act_neur].mean(), csi[act_neur].mean(), bsi[act_neur].mean()

def compute_statistics(track_statistics, net, eval_loaders, device, dataset, crit, q=2, seed=0, attribution_method="deeplift", remove_dead_units=True, print_statistics=False, exclude_1_dim_pars=True):
    # TODO: Dissociate num_samples from batch size arguments.
    statistics = {}
    net.eval()
    layers = get_layers(net, dream_team=False, as_numpy=True)
    if exclude_1_dim_pars:
      layers = [layer for layer in layers if len(layer.shape) > 1]
    num_classes = list(net.parameters())[-1].shape[0]
    for statistic in [stat for stat in track_statistics if "net_comp" in stat]:
      print(statistic) if print_statistics else None
      kappa = float(statistic.split("_")[-1]) # e.g. net_comp_0.1
      statistics[statistic] = get_network_compressibility(layers, kappa=kappa, q=q)
    for statistic in [stat for stat in track_statistics if "par_comp" in stat]:
      print(statistic) if print_statistics else None
      kappa = float(statistic.split("_")[-1]) # e.g. par_comp_0.1
      statistics[statistic] = np.array([get_compressibility(layer, kappa=kappa, q=q, flat="flat" in statistic) for layer in layers])
    for statistic in [stat for stat in track_statistics if "par_phdim" in stat]:
      print(statistic) if print_statistics else None
      k = int(statistic.split("_")[-1]) # e.g. par_phdim_128
      statistics[statistic] = np.array([get_ph_dim(layer, k) for layer in layers])
    for statistic in [stat for stat in track_statistics if "act_phdim" in stat]:
      print(statistic) if print_statistics else None
      _, _, split, layer_idx, num_samples, k = statistic.split("_") # e.g. act_phdim_train_m2_1000_128
      k = int(statistic.split("_")[-1]) 
      layer_idx = int(layer_idx.replace("m", "-"))
      num_samples = int(num_samples)

      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      images, labels = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device)
      z, y, y_hat = get_representations(deepcopy(net).to(device), layer_idx, images, labels, data_loader.batch_size)
      statistics[statistic] = get_ph_dim(z, k)
    for statistic in [stat for stat in track_statistics if "eff-bcr-nll" in stat]:
      _, split, num_samples = statistic.split("_") # example: eff-bcr-nll_train_1000
      assert "nll" in statistic
      num_samples = int(num_samples)
      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      nll = nn.CrossEntropyLoss(reduction='none').to(device)
      try:
        X, y, s = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device, bias=True)
        is_spur_dataset = True
      except: # for datasets w/o bias labels.
        X, y = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device, bias=False)
        is_spur_dataset = False
      # net_stat = torch.compile(net, backend='nvfuser').to(device)
      # try:
      #   net_stat.train()
      #   with torch.no_grad():  
      #     out = net_stat(X)
      #   net_stat.eval()
      try:
        net.train()
        with torch.no_grad():  
          out = net(X)
        net.eval()
        loss = nll(out, y)
        if is_spur_dataset:
          statistics[statistic] = (loss[y != s].sum() / loss.sum()).item()
          statistics[statistic + "num"] = loss[y != s].sum().item()
          statistics[statistic + "den"] = loss.sum().item() 
          statistics[statistic + "_bcba"] = statistics[statistic + "num"] / (statistics[statistic + "den"] - statistics[statistic + "num"])
        else:
          statistics[statistic] = statistics[statistic + "num"] = statistics[statistic + "den"] = np.nan
        softmaxes, _ = torch.exp(out - torch.logsumexp(out, dim=1, keepdim=True)).max(dim=1)
        loss_norm = loss / loss.sum()
        statistics[statistic.replace("eff-bcr-nll", "loss_entropy")] = -torch.nansum(loss_norm * torch.log(loss_norm)).item()
        statistics[statistic.replace("eff-bcr-nll", "max_softmax")] = torch.nanmean(softmaxes).item()
        statistics[statistic.replace("eff-bcr-nll", "logits_norm")] = out.norm(dim=1).mean().item()
        statistics[statistic.replace("eff-bcr-nll", "parameters_norm")] = torch.stack([p.data.norm() for p in net.parameters()]).cpu().numpy()
      except:
        if is_spur_dataset:
          statistics[statistic] = statistics[statistic + "num"] = statistics[statistic + "den"] = np.nan
        else:
          statistics[statistic] = statistics[statistic + "num"] = statistics[statistic + "den"] = np.nan
        statistics[statistic.replace("eff-bcr-nll", "loss_entropy")] = np.nan
        statistics[statistic.replace("eff-bcr-nll", "max_softmax")] = np.nan
        statistics[statistic.replace("eff-bcr-nll", "logits_norm")] = np.nan
        statistics[statistic.replace("eff-bcr-nll", "parameters_norm")] = np.nan
      
    for statistic in [stat for stat in track_statistics if "csi" in stat]:
      _, split, layer_idx, num_samples = statistic.split("_")
      layer_idx = int(layer_idx.replace("m", "-"))
      num_samples = int(num_samples)
      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      images, labels, spurs = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device, bias=True)
      # ts = get_timestamp()
      # torch.save(net, f"results/debug/net_{ts}.pyT")
      # net = torch.load(f"results/debug/net_{ts}.pyT")
      # os.remove(f"results/debug/net_{ts}.pyT")
      try:
        z, y, y_hat, s = get_representations(deepcopy(net), layer_idx, images, labels, data_loader.batch_size, spurs=spurs)
        z = np.reshape(z, (z.shape[0], -1))      
        z = np.abs(z)
      except:
        # debug_file = f"results/debug_{get_timestamp()}.pyT"
        # torch.save({
        #   "net": deepcopy(net),
        #   "images": images,
        #   "labels": labels,
        #   "spurs": spurs
        # }, debug_file)
        # raise Exception
        z = np.zeros((20, 10))
        y = y_hat = s = np.zeros((20))
        z[:, :] = np.nan
        y[:] = np.nan
        
      if "biasa" in statistic:
        idx = y == s
        z, y, y_hat, s = z[idx], y[idx], y_hat[idx], s[idx]
      if "biasc" in statistic:
        idx = y != s
        z, y, y_hat, s = z[idx], y[idx], y_hat[idx], s[idx] 

      class_mean_activations = np.stack([z[y==i].mean(0) for i in range(num_classes)]).T
      csi, max_idx = np.apply_along_axis(partial(compute_class_selectivity_index, return_max_idx=True), 1, class_mean_activations).T

      if "weighted" in statistic:
        # class_ratio_activations = np.stack([(z[y==i] > 0).mean(0) for i in range(num_classes)]).T        
        # csi = csi * class_ratio_activations[np.arange(len(max_idx)), max_idx.astype(int)]
        # statistics[statistic] = csi.mean() / (z > 0).mean()
        activations_mean = class_mean_activations.mean(1)
        statistics[statistic] = np.array([np.linalg.norm(class_mean_activations[:, i]-activations_mean)  for i in range(num_classes)]).mean()/np.linalg.norm(activations_mean)
      else:
        csi_nz = csi[csi != 0]
        statistics[statistic] = csi_nz.mean()
    
      bias_mean_activations = np.stack([z[s==i].mean(0) for i in range(num_classes)]).T
      bsi, max_idx = np.apply_along_axis(partial(compute_class_selectivity_index, return_max_idx=True), 1, bias_mean_activations).T
      if "weighted" in statistic:
        # bias_ratio_activations = np.stack([(z[s==i] > 0).mean(0) for i in range(num_classes)]).T        
        # bsi = bsi * bias_ratio_activations[np.arange(len(max_idx)), max_idx.astype(int)]
        # statistics[statistic.replace("csi", "bsi")] = bsi.mean()
        activations_mean = bias_mean_activations.mean(1)
        statistics[statistic.replace("csi", "bsi")] = np.array([np.linalg.norm(bias_mean_activations[:, i]-activations_mean)  for i in range(num_classes)]).mean()/np.linalg.norm(activations_mean)
      else:
        bsi_nz = bsi[bsi != 0]
        statistics[statistic.replace("csi", "bsi")] = bsi_nz.mean()

      if "weighted" in statistic:
        group_mean_activations = np.stack([z[(y==i) & (s==j)].mean(0) for i in range(num_classes) for j in range(num_classes)]).T
        activations_mean = group_mean_activations.mean(1)

        statistics[statistic.replace("csi", "gsi")] = np.array([np.linalg.norm(group_mean_activations[:, i]-activations_mean)  for i in range(num_classes)]).mean()/np.linalg.norm(activations_mean)

        csi_gt_bsi_ratio, avg_csi_in_bias, active_neurons_ratio, activation_ratio, avg_csi_in_tobias, *mean_values = compute_avg_csi_in_bias(z, y, s, num_classes, active_neurons_threshold=0.0)
        statistics[statistic.replace("-weighted", "").replace("csi", "csi-gt-bsi-ratio")] = csi_gt_bsi_ratio
        statistics[statistic.replace("-weighted", "").replace("csi", "avg-csi-in-bias")] = avg_csi_in_bias
        statistics[statistic.replace("-weighted", "").replace("csi", "csi-act-neur-ratio")] = active_neurons_ratio
        statistics[statistic.replace("-weighted", "").replace("csi", "csi-act-ratio")] = activation_ratio
        statistics[statistic.replace("-weighted", "").replace("csi", "avg-csi-in-tobias")] = avg_csi_in_tobias
        weighted_csi, weighted_bsi, csi, bsi = mean_values
        statistics[statistic.replace("-weighted", "-weighted-avg")] = weighted_csi
        statistics[statistic.replace("-weighted", "-avg")] = csi
        statistics[statistic.replace("-weighted", "-weighted-avg").replace("csi", "bsi")] = weighted_bsi
        statistics[statistic.replace("-weighted", "-avg").replace("csi", "bsi")] = bsi

    for statistic in [stat for stat in track_statistics if "csep" in stat]:
      _, split, layer_idx, num_samples = statistic.split("_")
      layer_idx = int(layer_idx.replace("m", "-"))
      num_samples = int(num_samples)
      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      images, labels, spurs = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device, bias=True)
      try:
        z, y, y_hat, s = get_representations(deepcopy(net), layer_idx, images, labels, data_loader.batch_size, spurs=spurs, eval_mode="-eval" in statistic)
        z = np.reshape(z, (z.shape[0], -1))      
        z = np.abs(z)
      except:
        z = np.zeros((20, 10))
        y = y_hat = s = np.zeros((20))
        z[:, :] = np.nan
        y[:] = np.nan

      # max_values = z.max(axis=0, keepdims=True)
      # max_values[max_values == 0] = 1
      # z = z / max_values
      # z -= z.mean(axis=0, keepdims=True)
      # std = z.std(axis=0, keepdims=True)
      # std[std == 0] = 1
      # z = z / std
      # z = z[:, z.sum(0) > 10]

      similarity = "l1" if "l1" in statistic else "cosine"

      if "biasa" in statistic:
         idx = y == s
         z, y, y_hat, s = z[idx], y[idx], y_hat[idx], s[idx]
      if "biasc" in statistic:
         idx = y != s
         z, y, y_hat, s = z[idx], y[idx], y_hat[idx], s[idx] 

      statistics[statistic], statistics[statistic + "_dw"], statistics[statistic + "_db"] = get_class_separation(torch.tensor(z).to(device), torch.tensor(y).to(device), num_classes, similarity=similarity)
      statistics[statistic.replace("csep", "bsep")], statistics[statistic.replace("csep", "bsep") + "_dw"], statistics[statistic.replace("csep", "bsep") + "_db"] = get_class_separation(torch.tensor(z).to(device), torch.tensor(s).to(device), num_classes, similarity=similarity, median="med" in statistic)

    for statistic in [stat for stat in track_statistics if "deadunit" in stat]:
      print(statistic) if print_statistics else None
      stat_type, split, layer_idx = statistic.split("_") # e.g. act_phdim_train_m2_1000_128
      layer_idx = int(layer_idx.replace("m", "-"))

      if "-" in stat_type:
          _, atol = stat_type.split("-")
          atol = float(atol)
      else:
          atol = 0.0

      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      images, labels = get_samples_from_loader(data_loader, len(data_loader.dataset), seed=seed, device=device)
      z, y, y_hat = get_representations(deepcopy(net).to(device), layer_idx, images, labels, data_loader.batch_size)


      width = z.shape[1]
      statistics[statistic] = ((~test_equality(z, 0, atol)).sum(0) == 0).sum()
      statistics[statistic + "_cc"] = np.array([((~test_equality(z[y == cls], 0, atol)).sum(0) == 0).sum() for cls in range(num_classes)], dtype=float)
      statistics[statistic + "_ccavg"] = statistics[statistic + "_cc"].mean()

      statistics[statistic + "_rel"] = statistics[statistic] / width
      statistics[statistic + "_rel_cc"] = statistics[statistic + "_cc"] / np.array([z[y == cls].shape[1] for cls in range(num_classes)])
      statistics[statistic + "_rel_ccavg"] = statistics[statistic + "_rel_cc"].mean()
      for statistic_2 in [stat for stat in track_statistics if ("effcomp" in stat) and (split in stat)]:
        print(statistic_2) if print_statistics else None
        kappa = float(statistic_2.split("_")[-1]) # e.g. flat_effcomp_train_0.1 or
        layer_idx = int(statistic_2.split("_")[-2])
        mask = z.sum(0) == 0
        statistics[statistic_2] = np.array([
           get_compressibility(jnp.array(layer).at[mask].set(0.0) if i == layer_idx else jnp.array(layer), kappa=kappa, q=q, flat="flat" in statistic_2) for i, layer in enumerate(layers)
           ])   
    if "par_er" in track_statistics:
      print("par_er") if print_statistics else None
      statistics["par_er"] = np.array([get_effective_rank(layer) for layer in layers])
    for statistic in [stat for stat in track_statistics if "classcorr" in stat]:
      print(statistic) if print_statistics else None
      _, split, layer_idx, num_samples = statistic.split("_")
      layer_idx = int(layer_idx.replace("m", "-"))
      num_samples = int(num_samples)
      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      images, labels = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device)
      z, y, y_hat = get_representations(deepcopy(net), layer_idx, images, labels, data_loader.batch_size)
      
      if remove_dead_units:
        z = z[:, z.sum(0) != 0]
      
      df = pd.DataFrame(columns=range(num_classes))
      for i in range(num_classes):
          df[i] = z[y == i].mean(0).flatten()
      if "l1" in statistic:
        v = torch.tensor(df.values.T)
        # v_normalized = v / v.norm(p=1, dim=1, keepdim=True)
        c = 1 - torch.abs(v.unsqueeze(1) - v.unsqueeze(0)).sum(dim=2)/torch.abs(v).sum(1).mean()
        c = c.detach().cpu().numpy()
      else:
        c = df.corr().values
      statistics[statistic] = c
      statistics[statistic + "_avg"] = c[c!=1].mean()
      if "l1" not in statistics:
        df_binary = df  > 0
        m = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                m[i, j] = (df_binary[i] == df_binary[j]).mean()
        statistics[statistic + "_bin"] = m
        statistics[statistic + "_bin" + "_avg"] = m[m!=1].mean() if (m!=1).any() else np.nan

    for statistic in [stat for stat in track_statistics if "layercond" in stat]:
      print(statistic) if print_statistics else None
      _, split, layer_idx, kappa, num_samples = statistic.split("_") # example layercond_train_m2_0.05_1000
      kappa = float(kappa)
      layer_idx = int(layer_idx.replace("m", "-"))
      num_samples = int(num_samples)
      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      images, labels = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device)
      class_conds = get_layer_conductance(deepcopy(net), images, labels, num_classes, layer_idx, kappa)
      statistics[statistic] = class_conds.max(1).mean()
      statistics[statistic + "_cc"] = np.array([class_conds[np.argmax(class_conds, 1) == i].max(1).mean() if (np.argmax(class_conds, 1) == i).sum() > 0 else np.nan for i in range(num_classes)])
      statistics[statistic + "_clcounts"] = np.array([(np.argmax(class_conds, 1) == i).sum() for i in range(num_classes)])
    for statistic in [stat for stat in track_statistics if "act_er" in stat]:
      print(statistic) if print_statistics else None
      _, _, split, layer_idx, num_samples = statistic.split("_") # e.g. act_er_train_m2_1000
      layer_idx = int(layer_idx.replace("m", "-"))
      num_samples = int(num_samples)
      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      images, labels, spurs = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device, bias=True)
      z, y, y_hat, s = get_representations(deepcopy(net), layer_idx, images, labels, data_loader.batch_size, spurs=spurs)
      z = np.reshape(z, (z.shape[0], -1))
      statistics[statistic] = get_effective_rank(z)
      statistics[statistic + "_cc"] = np.array([get_effective_rank(z[y == cls]) for cls in range(num_classes)])
      statistics[statistic + "_ccavg"] = np.mean(statistics[statistic + "_cc"])
      statistics[statistic + "_bc"] = np.array([get_effective_rank(z[s == cls]) for cls in range(num_classes)])
      statistics[statistic + "_bcavg"] = np.mean(statistics[statistic + "_bc"])
      del z
    for statistic in [stat for stat in track_statistics if "act_sr" in stat]:
      print(statistic) if print_statistics else None
      _, _, split, layer_idx, num_samples = statistic.split("_") # e.g. act_er_train_m2_1000
      layer_idx = int(layer_idx.replace("m", "-"))
      num_samples = int(num_samples)
      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      images, labels = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device)
      z, y, y_hat = get_representations(deepcopy(net), layer_idx, images, labels, data_loader.batch_size)
      statistics[statistic] = get_stable_rank(z)
      statistics[statistic + "_cc"] = np.array([get_stable_rank(z[y == cls]) for cls in range(list(net.parameters())[-1].shape[0])])
      statistics[statistic + "_ccavg"] = np.mean(statistics[statistic + "_cc"])
      del z
    for statistic in [stat for stat in track_statistics if "act_pqi" in stat]:
      print(statistic) if print_statistics else None
      _, pqi, split, layer_idx, num_samples = statistic.split("_") # e.g. 
      _, p, q = pqi.split("-")
      p, q = float(p), float(q)
      layer_idx = int(layer_idx.replace("m", "-"))
      num_samples = int(num_samples)
      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      if "-bc" in statistic:
        images, labels, spurs = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device, bias=True)
        z, y, y_hat, s = get_representations(deepcopy(net), layer_idx, images, labels, data_loader.batch_size, spurs=spurs)
        z, y, y_hat, s = z[y!=s], y[y!=s], y_hat[y!=s], s[y!=s] 
      else:
        images, labels = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device)
        z, y, y_hat = get_representations(deepcopy(net), layer_idx, images, labels, data_loader.batch_size)
      z = np.reshape(z, (z.shape[0], -1))
      statistics[statistic] = get_pqi(z, p, q, int(z.shape[1])).mean()
      del z
    for statistic in [stat for stat in track_statistics if "act_comp" in stat]:
      eval_mode = False
      print(statistic) if print_statistics else None
      _, _, split, layer_idx, num_samples, kappa = statistic.split("_") # e.g. act_comp_train_m2_1000_0.1
      kappa = float(kappa) 
      layer_idx = int(layer_idx.replace("m", "-"))
      num_samples = int(num_samples)
      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      if "-bc" in statistic:
        images, labels, spurs = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device, bias=True)
        z, y, y_hat = get_representations(deepcopy(net), layer_idx, images, labels, data_loader.batch_size, spurs=spurs, eval_mode=eval_mode)
        z, y, y_hat = z[y!=s], y[y!=s], y_hat[y!=s], s[y!=s] 
      else:
        images, labels = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device)
        z, y, y_hat = get_representations(deepcopy(net), layer_idx, images, labels, data_loader.batch_size, eval_mode=eval_mode)
      # TODO: Remove the effect of having had logits here
      statistic_sp = statistic.replace("comp", "sp")
      statistic_sp = "_".join(statistic_sp.split("_")[:-1])
      statistics[statistic] = get_compressibility(z, kappa=kappa, q=q, flat=False)
      statistics[statistic.replace("comp", "norm")] = np.linalg.norm(z, axis=1).mean()
      statistics[statistic_sp] = (z == 0).mean()
      statistics[statistic + "_cc"] = np.array([get_compressibility(z[y == cls], kappa=kappa, q=q, flat=False) for cls in range(list(net.parameters())[-1].shape[0])])
      statistics[statistic_sp + "_cc"] = np.array([(z[y == cls] == 0).mean() for cls in range(list(net.parameters())[-1].shape[0])])
      statistics[statistic + "_ccavg"] = np.mean(statistics[statistic + "_cc"])
      statistics[statistic_sp + "_ccavg"] = np.mean(statistics[statistic + "_cc"])
      # statistics[f"cc_accuracy_{split}_{num_samples}"] = np.array([(y[y == i] == y_hat[y == i]).mean() for i in range(num_classes)])
      # statistics[f"accuracy_{split}_{num_samples}"] = (y == y_hat).mean()
      del z
    for statistic in [stat for stat in track_statistics if (("accuracy" in stat) and ("abratio" not in stat))]:
      acc_statistic, split, num_samples = statistic.split("_")
      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      try:
        idx, images, y, b = get_samples_from_loader(data_loader, int(num_samples), seed=seed, device=device, bias=True, get_idx=True)
        is_bd = True
      except:
        # HACK for datasets that do not have bias.
        is_bd = False
        idx, images, y = get_samples_from_loader(data_loader, int(num_samples), seed=seed, device=device, bias=False, get_idx=True)
        b = y
      # FIXED: 2024-09-14
      try:
      # HACK
        y_hat, logits, losses = get_output(net, images, y, data_loader.batch_size, device=device, spurs=None, eval_mode="-eval" in statistic, criterion="nll" if ("accuracy-full" in statistic) and isinstance(crit, nn.CrossEntropyLoss) else None)
      except Exception as error:
        print(error)
        y_hat = torch.tensor(np.zeros(len(b))).to(device)
        y_hat[:] = np.nan
        losses = y_hat
      # if statistic == "accuracy_train_10000":
      #   np.save(idx.detach().cpu().numpy(), f"results/{dataset}_{statistic}_idx.npy")
      #   raise Exception # HACK
      statistics[f"cc_{acc_statistic}_{split}_{num_samples}"] = torch.tensor([(y[y == i] == y_hat[y == i]).float().mean() for i in range(num_classes)]).cpu().numpy()
      statistics[f"{acc_statistic}_{split}_{num_samples}"] = (y == y_hat).float().mean().cpu().numpy().item() 
      statistics[f"{acc_statistic}_bias_{split}_{num_samples}"] = (b == y_hat).float().mean().cpu().numpy().item() if is_bd else np.nan
      statistics[f"{acc_statistic}_ba_{split}_{num_samples}"] = (y[y==b] == y_hat[y==b]).float().mean().cpu().numpy().item()  if is_bd else np.nan
      statistics[f"{acc_statistic}_bc_{split}_{num_samples}"] = (y[y!=b] == y_hat[y!=b]).float().mean().cpu().numpy().item()  if is_bd else np.nan
      # How much of the missed predictions in bias-conflicting examples predicted in line with the bias signal?
      statistics[f"{acc_statistic}_bbc_{split}_{num_samples}"] = (b[(y!=b) & (y!=y_hat)] == y_hat[(y!=b) & (y!=y_hat)]).float().mean().cpu().numpy().item()  if is_bd else np.nan
      statistics[f"{acc_statistic}_cb_matrix_{split}_{num_samples}"] = np.reshape(np.array([(y[(y==i) & (b==j)] == y_hat[(y==i) & (b==j)]).detach().cpu().numpy().mean() for i in range(num_classes) for j in range(num_classes)]), (num_classes, num_classes))  if is_bd else np.nan
      if "-full" in statistic:
        statistics[f"{acc_statistic.replace('accuracy', 'idx')}_y_s_yhat_{split}_{num_samples}"] = torch.vstack((idx[torch.argsort(idx)], y[torch.argsort(idx)], b[torch.argsort(idx)], y_hat[torch.argsort(idx)])).detach().cpu().numpy()
        statistics[f"{acc_statistic.replace('accuracy', 'losses')}_{split}_{num_samples}"] = losses[torch.argsort(idx)].detach().cpu().numpy()
      else:
        statistics[f"{acc_statistic.replace('accuracy', 'idx')}_mispred_{split}_{num_samples}"] = idx[y!=y_hat].detach().cpu().numpy()
        statistics[f"{acc_statistic.replace('accuracy', 'idx')}_y_s_yhat_mispred_{split}_{num_samples}"] = torch.vstack((idx, y, b, y_hat)).T[y!=y_hat].detach().cpu().numpy()

    for statistic in [stat for stat in track_statistics if "abratio" in stat]:
      _, metric, split, num_samples = statistic.split("_") # abratio_accuracy_train_1000
      statistics[f"abratio_cc_{metric}_{split}_{num_samples}"] = statistics[f"cc_{metric}_ab{split}_{num_samples}"] / statistics[f"cc_{metric}_{split}_{num_samples}"]
      statistics[f"abratio_{metric}_{num_samples}"] = statistics[f"{metric}_ab{split}_{num_samples}"] / statistics[f"{metric}_{split}_{num_samples}"]
    for statistic in [stat for stat in track_statistics if "attr" in stat]:
      print(statistic) if print_statistics else None
      # HACK
      ts = get_timestamp()
      torch.save(net, f"results/debug/net_{ts}.pyT")
      net = torch.load(f"results/debug/net_{ts}.pyT")
      os.remove(f"results/debug/net_{ts}.pyT")
      net.train()
      attr_name = statistic.split("_")[0]
      if "-" not in attr_name:
        attribution_method = attribution_method
      else:
        attribution_method = attr_name.split("-")[1]
      split = statistic.split("_")[-2]
      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      num_samples = int(statistic.split("_")[-1]) # e.g. attr_train_1000
      num_samples = num_samples if num_samples else data_loader.batch_size
      images, labels = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device)
      images, labels = images[:num_samples], labels[:num_samples]
      # print(images, labels)
      # images = images.float()
      images.requires_grad = True
      # labels.requires_grad = True
      attribution_function =  get_attribution_function(attribution_method)
      af = attribution_function(net)
      if "doublemnist" in dataset:
        d = images.shape[-2]
        spurious_features_mask = torch.hstack((torch.zeros(d, d, dtype=bool), torch.ones(d, d, dtype=bool)))
      elif ("wrane" in dataset) or ("wrale" in dataset) or ("wxane" in dataset) or ("wxale" in dataset):
        _, color_width, window_width = dataset.split("_")[-2].split("-")
        color_width, window_width = int(color_width), int(window_width)
        if "attribute" in statistic: 
          spurious_features_mask = get_wrane_mask(images[0, 0].shape, color_width, window_width)
        else:
          spurious_features_mask_all = (images.sum(1) == images[:, :, window_width-1, window_width-1].sum(1)[:, np.newaxis, np.newaxis])[:, np.newaxis, :, :]
      else:
        raise Exception
        # spurious_features_mask_all = (images.sum(1) == images[:, :, 0, 0].sum(1)[:, np.newaxis, np.newaxis])[:, np.newaxis, :, :]


      color_attrs_exp = np.zeros(num_samples)
      empty_attrs_exp = np.zeros(num_samples)
      for ind in range(num_samples):
          input = images[ind].unsqueeze(0).to(device)
          # input.requires_grad = True
          # if "doublemnist" in dataset:
          #   spurious_features_mask = spurious_features_mask_all[0, 0]
          # else:
          #   spurious_features_mask = spurious_features_mask_all[ind][0]

          if (("wrane" in dataset) or ("wrale" in dataset)) and ("attribute" not in statistic):
            spurious_features_mask = spurious_features_mask_all[ind][0]
          # print(spurious_features_mask.shape)
          # if "wrale" in dataset:
          #     spurious_features_mask_wrale = partition_mask(spurious_features_mask.cpu().detach().numpy(), labels[ind])
          net.zero_grad()
          try:
            attrs = attribute_image_features(af, net, input, target=labels[ind], baselines=input * 0)
          except TypeError:
            attrs = af.attribute(input, target=labels[ind])
          attrs = np.transpose(attrs.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
          try:
            _attr = viz._normalize_attr(attrs + 1e-6, "absolute_value", 2, reduction_axis=2)
            # if "wrale" in dataset:
            #   color_attrs_exp[ind] = _attr[spurious_features_mask_wrale].sum()/_attr.sum()
            # else:
            #   color_attrs_exp[ind] = _attr[spurious_features_mask.cpu().detach().numpy()].sum()/_attr.sum()
            color_attrs_exp[ind] = _attr[spurious_features_mask.cpu().detach().numpy()].sum()/_attr.sum()
            empty_attrs_exp[ind] = _attr[(input[0].sum(0)==0).cpu().detach().numpy()].sum()/_attr.sum()
          except:
            print(f"Warning! Problem with computing attributions for image {ind}.")
            # print(attrs)
            _attr = viz._normalize_attr(attrs, "absolute_value", 2, reduction_axis=2)
            color_attrs_exp[ind] = np.nan
            empty_attrs_exp[ind] = np.nan
      net.eval()
      statistics[statistic] = np.nanmean(color_attrs_exp)
      # print(color_attrs_exp)
      # print(labels.cpu().detach().numpy())
      statistics[statistic + "_cc"] = np.array([np.nanmean(color_attrs_exp[labels.cpu().detach().numpy() == i]) for i in range(num_classes)])
      statistics[statistic.replace("attr", "ettr")] = np.nanmean(empty_attrs_exp)
    for statistic in [stat for stat in track_statistics if "hessian" in stat]:
      print(statistic) if print_statistics else None
      split = statistic.split("_")[-2]
      data_loader = eval_loaders[split.replace("biasedval", "biased_val").replace("rand", "_rand").replace("wgtest", "wg_test").replace("eqtest", "eq_test")]
      num_samples = int(statistic.split("_")[-1]) # e.g. hessian_train_1000
      num_samples = num_samples if num_samples else data_loader.batch_size
      images_hessian, labels = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device)
      if "frame" in dataset:
        images_hessian[:, :, 1:-1, 1:-1] = 0.
      elif ("wrame" in dataset):
        window_width = int(dataset.split("_")[-2].split("-")[-1])
        images_hessian[:, :, window_width:-window_width, window_width:-window_width] = 0.
      elif "background" in dataset:
        images_hessian[:, :, :, :] = images_hessian[:, :, 0, 0][:, :, np.newaxis, np.newaxis]
      elif "doublemnist" in dataset:
        d = images.shape[-2]
        images_hessian[:, :, :d, :] = 0.
      elif ("wrane" in dataset) or ("wrale" in dataset) or ("wxane" in dataset) or ("wxale" in dataset):
        _, color_width, window_width = dataset.split("_")[-2].split("-")
        color_width, window_width = int(color_width), int(window_width)
        spurious_features_mask = get_wrane_mask(images_hessian[0, 0].shape, color_width, window_width)
        images_hessian[:, :, ~spurious_features_mask] = 0.
      else:
          raise Exception
      with torch.random.fork_rng():
        # Set a local seed
        torch.manual_seed(seed)
        top_eigvals, _ = hessian(net, crit, data=(images_hessian, labels), cuda=True).eigenvalues()
      statistics[statistic] = top_eigvals[-1]
    return statistics

def get_max_and_rest_avg(arr):
    max = np.max(arr)
    max_idx = np.argmax(arr)
    rest_mean = np.mean([a for i, a in enumerate(arr) if i != max_idx])
    return max, max_idx, rest_mean

def compute_class_selectivity_index(arr, eps=1e-6, return_max_idx=False):
    max, max_idx, rest_mean = get_max_and_rest_avg(arr)
    if return_max_idx:
      return (max - rest_mean) / (max + rest_mean + eps), max_idx
    return (max - rest_mean) / (max + rest_mean + eps)

def partition_mask(mask, class_index, seed=42):
    """
    Partitions the pixels in the mask between 10 classes, sets the pixels
    associated with the specified class index to True, and the rest to False.
    
    :param mask: 2D boolean numpy array representing the mask.
    :param class_index: int, the class index for which pixels should be set to True.
    :param seed: int, seed for random number generator for consistent partitioning.
    :return: 2D boolean numpy array with the partitioned mask.
    """
    np.random.seed(seed)
    true_indices = np.argwhere(mask)
    np.random.shuffle(true_indices)
    n = len(true_indices) // 10  # Ensure division into 10 parts; discard extras
    
    # Calculate start and end indices for the class partition
    start_idx = class_index * n
    end_idx = (class_index + 1) * n if class_index < 9 else len(true_indices)
    
    # Initialize a false mask to populate selected class indices
    partitioned_mask = np.zeros_like(mask, dtype=bool)
    selected_indices = true_indices[start_idx:end_idx]
    partitioned_mask[selected_indices[:,0], selected_indices[:,1]] = True
    
    return partitioned_mask



def get_model_details(folder, get_model=True, model_iter=-1, bias=True, get_hist=True, get_data_loaders=True, get_samples=True, get_repr=True, split="test", num_samples=1000, layer_idx=-1, seed=None, device=None, repr_eval_mode=False):
    r = {}
    folder = folder + "/" if folder[-1] != "/" else folder
    r["args"] = args = torch.load(folder + "args.info")
    seed = args.seed if not seed else seed
    device = args.device if not device else device
    if get_hist:
        r["eval_hist"] = torch.load(folder + "evaluation_history.hist")
        r["train_hist"] = torch.load(folder + "training_history.hist")
    if get_model:
        r["net"] = net = torch.load(folder + "net.pyT") if model_iter == -1 else torch.load(folder + f"training_iters/net_{model_iter}.pyT")
        net.eval()
    if get_data_loaders:
        r["train_loader"], r["eval_loaders"], r["num_classes"], r["input_dim"] = get_data(args, return_biased=bias, abtrain=False)
    if get_samples:
        data_loader = r["eval_loaders"][split]
        idx, images, labels, *spurs = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device, bias=bias, get_idx=True)
        spurs = spurs[0] if bias else None
        r["idx"], r["images"], r["labels"], r["spurs"] = idx, images, labels, spurs
    if get_repr:    
        z, y, y_hat, *b = get_representations(deepcopy(net).to(device), layer_idx, images, labels, data_loader.batch_size, spurs=spurs, eval_mode=repr_eval_mode)
        z = np.reshape(z, (z.shape[0], -1))      
        z = np.abs(z)
        b = b[0] if bias else None
        r["y"], r["y_hat"], r["b"], r["z"] = y, y_hat, b, z
    return r