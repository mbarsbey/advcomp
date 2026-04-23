import torch
import torch.nn.functional as F
import numpy as np
import heapq
import os
import pandas as pd
from .common import get_model_info, rng_state_summary
from .statistics import compute_statistics

def get_loss_and_accuracy(df, fi, folder):
    df.loc[fi, "training_loss"] = torch.load(folder + "evaluation_history_TRAIN.hist",map_location='cpu')[-1][1]
    df.loc[fi, "test_loss"] = torch.load(folder + "evaluation_history_TEST.hist",map_location='cpu')[-1][1]
    df.loc[fi, "loss_diff"] = np.abs(df.loc[fi, "test_loss"] - df.loc[fi, "training_loss"])
    df.loc[fi, "training_accuracy"] = torch.load(folder + "evaluation_history_TRAIN.hist",map_location='cpu')[-1][2]
    df.loc[fi, "test_accuracy"] =     torch.load(folder + "evaluation_history_TEST.hist",map_location='cpu')[-1][2]
    df.loc[fi, "accuracy_diff"] = np.abs(df.loc[fi, "test_accuracy"] - df.loc[fi, "training_accuracy"])
    df.loc[fi, "training_loss_avg"] = torch.load(folder + "evaluation_history_AVGTRAIN.hist",map_location='cpu')[-1][1]
    df.loc[fi, "test_loss_avg"] = torch.load(folder + "evaluation_history_AVG.hist",map_location='cpu')[-1][1]
    df.loc[fi, "loss_diff_avg"] = np.abs(df.loc[fi, "test_loss_avg"] - df.loc[fi, "training_loss_avg"])
    df.loc[fi, "training_accuracy_avg"] = torch.load(folder + "evaluation_history_AVGTRAIN.hist",map_location='cpu')[-1][2]
    df.loc[fi, "test_accuracy_avg"] =     torch.load(folder + "evaluation_history_AVG.hist",map_location='cpu')[-1][2]
    df.loc[fi, "accuracy_diff_avg"] = np.abs(df.loc[fi, "test_accuracy_avg"] - df.loc[fi, "training_accuracy_avg"])
    
def accuracy(out, y):
    y = y.flatten()
    _, pred = out.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().float() / y.size(0)

def get_bias_tensor(y, y_hat, b, num_classes):
    t = np.zeros((num_classes, num_classes, num_classes))
    for i in range(len(y)):
        t[y[i], y_hat[i], b[i]] += 1
    return t

# @torch.compile
def evaluate(eval_loader, net, crit, args, print_std=True, bias_breakdown=False, num_samples=0, get_mispreds=False):
    num_total_samples = len(eval_loader.dataset) if hasattr(eval_loader, "dataset") else eval_loader.num_total_samples
    net.eval()
    # run over both test and train set
    total_size = 0
    total_loss = 0
    all_y = [] 
    all_b = []
    all_y_hat = []
    outputs = []
    all_mispreds = []

    with torch.no_grad():
        # P = 0  # num samples / batch size
        cur_num_samples = 0
        for k, (idx, x, y, *b) in enumerate(eval_loader):
            # P += 1
            # loop over dataset
            x, y = x.to(args.device), y.to(args.device)
            if "bert" in args.model:
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                out = net(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y
                )[1]
            else:
                out = net(x)
            if args.criterion.upper() == 'MSE':
                y = F.one_hot(y, num_classes=out.shape[1]).float()
            if k == 0:
                one_dim_output = out.shape[1] == 1
            _, y_hat = (None, (out > 0.5).float()) if one_dim_output else out.max(1)
            outputs.append(out)
            loss = crit(out, y)
            if get_mispreds:
                all_mispreds.append(idx[y!=y_hat])
            if getattr(args, "sample_weight", 1.0) != 1.0:
                loss = loss.mean()
            total_loss += float(loss) * x.size(0)
            
            all_y.append(y)
            all_y_hat.append(y_hat)
            if bias_breakdown:
                b = [b_.to(args.device) for b_ in b]
                all_b.extend(b)
            
            cur_num_samples += x.shape[0]
            if num_samples and (cur_num_samples > num_samples):
                break
            
        num_classes = out.shape[1]
        y = torch.concatenate(all_y)
        if args.criterion.upper() == 'MSE':
            _, y = y.max(1)
        y_hat = torch.concatenate(all_y_hat)
        if bias_breakdown:
            b = torch.concatenate(all_b)

    loss = total_loss / num_total_samples
    cc_accuracy = np.nan if one_dim_output else torch.tensor([(y[y == i] == y_hat[y == i]).float().mean() for i in range(num_classes)]).cpu().numpy() * 100
    accuracy = (y == y_hat).float().mean().cpu().numpy().item() * 100
    if bias_breakdown:
        accuracy_ba = (y[y==b] == y_hat[y==b]).float().mean().cpu().numpy().item() * 100 # bias_aligned
        accuracy_bc = (y[y!=b] == y_hat[y!=b]).float().mean().cpu().numpy().item() * 100 # bias_conflicting

    if print_std:
        print(loss, accuracy)

    if bias_breakdown:
        return {"loss": loss, "accuracy": accuracy, "cc_accuracy": cc_accuracy, "accuracy_ba": accuracy_ba, "accuracy_bc": accuracy_bc, "bias_tensor": get_bias_tensor(y, y_hat, b, num_classes), "mispreds": torch.concatenate(all_mispreds) if get_mispreds else np.nan}, outputs    
    return {"loss": loss, "accuracy": accuracy, "cc_accuracy": cc_accuracy, "mispreds": torch.concatenate(all_mispreds) if get_mispreds else np.nan}, outputs
    

### RESULTS FOLDER PROCESSING
def k_most_recently_modified(folders, k, modification_times):
    modification_times = [(os.path.getmtime(folder), folder) for folder in folders]
    most_recent = heapq.nlargest(k, modification_times)
    return [folder for _, folder in most_recent]

class ResultsFolder(object):
    def __init__(self, results_root_folder, k_most_recent=0, time_sorted=True, include_hist_folders=False):
        self.results_root_folder = results_root_folder
        self.timestamp, self.folders, self.results_summary_folder = prepare_result_analysis(results_root_folder, include_hist_folders=include_hist_folders)
        if k_most_recent or time_sorted:
            self.mod_times = [os.path.getmtime(folder + "/evaluation_history.hist") for folder in self.folders]
            time_idx = np.flip(np.argsort(self.mod_times))
            self.folders = np.array(self.folders)[time_idx].tolist()
            self.mod_times = np.array(self.mod_times)[time_idx].tolist()
            if k_most_recent > 0:
                self.folders, self.mod_times = self.folders[:k_most_recent], self.mod_times[:k_most_recent]
        self.exp_keys = [folder.split("/")[-2] for folder in self.folders]

def prepare_result_analysis(results_root_folder, include_hist_folders):
    assert (results_root_folder[-1] == "/") #and (results_root_folder != "results/")
    timestamp = results_root_folder.split("/")[1].split("_")[0];
    timestamp = timestamp if timestamp.isnumeric() else "00000000"
    if include_hist_folders:
        folders = sorted([r for r in [results_root_folder + f + "/" for f in os.listdir(results_root_folder)] if os.path.isdir(r) and os.path.exists(r + "evaluation_history.hist")])
    else:
        folders = sorted([r for r in [results_root_folder + f + "/" for f in os.listdir(results_root_folder)] if os.path.isdir(r) and os.path.exists(r + "net.pyT")])
    results_summary_folder = results_root_folder + f"{timestamp}_results_summary/"
    os.makedirs(results_summary_folder) if not os.path.exists(results_summary_folder) else None
    return timestamp, folders, results_summary_folder


def sample_result_folders(results_folder, no_samples):
    df = pd.read_csv(results_folder.results_summary_folder + results_folder.timestamp + "_x_mc_results.csv")
    df = df.sort_values("lr_b")
    chosen_folders = df.folder.iloc[list(map(int, np.linspace(0, len(df.folder)-1, no_samples)))].to_list()
    return [f for f in results_folder.folders if any([k + "/" in f for k in chosen_folders])]

@torch.compile
def evaluate_iteration(nets, eval_loaders, iteration, crit, args, print_std=False, batch_convergence=True, ignore_eval=False):
    num_classes =args.num_classes
    iterate_evals = {"iteration": iteration}
    iterate_outputs = {"iteration": iteration}
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        for net_prefix, eval_net in nets.items():
            for split, eval_loader in eval_loaders.items():
                if (args.check_batch_convergence) and (not args.legacy_evaluation) and (split == "train") and (iteration % args.num_improvement_iters !=0) and (iteration != args.iterations) and ( not batch_convergence):
                    eval_results, outputs = {"loss": np.nan, "accuracy": np.nan, "cc_accuracy": np.array([np.nan for _ in range(num_classes)])}, np.nan
                else:
                    if ignore_eval and (split != "train"):
                        eval_results, outputs = {"loss": np.nan, "accuracy": np.nan, "cc_accuracy": np.array([np.nan for _ in range(num_classes)])}, np.nan
                    else:
                        eval_results, outputs = evaluate(eval_loader, eval_net, crit, args, print_std=print_std)
                key = "eval_" + net_prefix.replace("net", "") + split
                iterate_outputs[key] = outputs
                for eval_key, eval_result in eval_results.items():                
                    iterate_evals[key + "_" + eval_key] = eval_result
            if "eval_abtrain_accuracy" in iterate_evals.keys():
                iterate_evals["eval_abratio_train_accuracy"] = iterate_evals["eval_abtrain_accuracy"] / iterate_evals["eval_train_accuracy"]
                iterate_evals["eval_abratio_train_cc_accuracy"] = iterate_evals["eval_abtrain_cc_accuracy"] / iterate_evals["eval_train_cc_accuracy"]
        iterate_statistics = compute_statistics(args.track_statistics, nets["net"], eval_loaders, args.device, args.dataset, crit, q=2, seed=args.seed, attribution_method=args.attribution_method)
        iterate_evals.update(iterate_statistics)
    return iterate_evals, iterate_outputs

def get_iterate_evaluation_msg(evals, eval_splits, train_b_eval={"loss": np.nan, "accuracy": np.nan}):
    # TODO: For now Ignoring avg. net evaluations. 
    msg = f"## Iteration {evals['iteration']}\n\n"
    eval_splits = ["train", "test"] + [split for split in eval_splits if split not in ["train", "test"]]
    msg += f"train batch:\n{train_b_eval['loss']:.6f}, {train_b_eval['accuracy']:.6f}\n"
    for split in eval_splits:
        msg += f"{split} split:\n"
        msg += f"{evals[f'eval_{split}_loss']:.6f}, {evals[f'eval_{split}_accuracy']:.6f}" 
        msg += "\n"
    msg += "statistics:\n"
    msg += str({key: item for key, item in evals.items() if (("eval" not in key) and ("iteration" not in key))})
    msg += "\n\n"
    return msg