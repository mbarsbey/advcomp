def main():
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=100000, type=int)
    parser.add_argument('--batch_size_train', '-b', default=100, type=int)
    parser.add_argument('--batch_size_eval', '--be', default=0, type=int,
                        help='must be equal to training batch size')
    parser.add_argument('--lr', nargs="*", type=float, default=[0.0, ])
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--momentum', '--mom', default=0, type=float)
    parser.add_argument('--print_freq', default=0, type=int)
    parser.add_argument('--eval_freq', default=200, type=int)
    parser.add_argument('--train_stats_freq', '--ts_freq', default=100, type=int)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--convergence_method', "--cm", type=str, choices=['custom', 'dataset', 'none'], default='dataset')
    parser.add_argument('--convergence_accuracy', "--conv_acc", type=float, default=0.)
    parser.add_argument('--convergence_loss', "--conv_loss", type=float, default=1000000.)
    parser.add_argument('--convergence_extra_iters', type=int, default=0)
    parser.add_argument('--num_improvement_iters', '--improve_iters', type=int, default=25000)
    parser.add_argument('--path', default='./data', type=str)
    parser.add_argument('--seed', default=0, type=int)  
    parser.add_argument('--model', default='fcn', type=str)
    parser.add_argument('--criterion', default='NLL', type=str,
                        help='NLL | linear_hinge')
    parser.add_argument('--width', default=100, type=int,
                        help='width of fully connected layers')
    parser.add_argument('--depth', default=2, type=int,
                        help='total number of hidden layers + input layer')
    parser.add_argument('--save_dir', default='results/', type=str)
    parser.add_argument('--lr_algorithm', default="sgd")
    parser.add_argument('--adam_b1', type=float, default=0.9)
    parser.add_argument('--adam_b2', type=float, default=0.999)
    parser.add_argument('--custom_init', type=str, default="")
    parser.add_argument('--traj', action='store_true', default=False)
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--schedule', action='store_true', default=False)
    parser.add_argument('--preprocess', action='store_true', default=False)
    parser.add_argument('--lr_scheduler', '--lrs', type=str, default="")
    parser.add_argument('--lrs_milestones', '--lrs_m', nargs="*", type=int, default=[])
    parser.add_argument('--lrs_gamma', '--lrs_g', type=float, default=0.1)
    parser.add_argument('--ignore_existing', '-i', action='store_true', default=False)
    parser.add_argument('--ignore_existing_data_cycle', '-id', action='store_true', default=False)
    parser.add_argument('--data_scale', "--ds", type=float, default=1.)
    parser.add_argument("--param_scale", type=float, default=0.0)
    parser.add_argument('--iter_record_freq', type=int, default=0, help="if this is 0, do nothing special. if this is >0, record model every n iterations. If this is -1, then follow the specified iter_record_freq scheme.")
    parser.add_argument('--script', '--experiment_script', '-s', nargs="+", type=str, default="")
    parser.add_argument('--track_statistics', nargs="*", type=str, default=[])
    parser.add_argument('--attribution_method', type=str, default="dl")
    parser.add_argument('--float64', action='store_true', default=False)
    parser.add_argument('--use_bias', '--bias', action='store_true', default=False)
    parser.add_argument('--exp_register_file', type=str, default="logs/exp_register.log")
    parser.add_argument('--continue_suffix', type=str, default="_contd")
    parser.add_argument('--data_seed', '--dataset_seed', type=int, default=-1)
    parser.add_argument("--init_model_path",type=str, default="")
    parser.add_argument('--wd', nargs="*", type=float, default=[0.0, ])
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--rowwise_l2', '--group_lasso', '--row2ln', type=float, default=0.0)
    parser.add_argument('--scale_invariant_row2ln', action="store_true", default=False)
    parser.add_argument('--nuclear_norm', "--nucnorm", type=float, default=0.0)
    parser.add_argument("--linear_head",action="store_true", default=False)
    parser.add_argument("--ablate_top_k_neurons", type=int, default=0)
    parser.add_argument("--activation_exponent", type=float, default=0.0)
    parser.add_argument("--dropout", "--do", type=float, default=0.0)
    parser.add_argument("--check_batch_convergence", "--batch_conv", action="store_true", default=False)
    parser.add_argument("--legacy_evaluation", action="store_true", default=False)
    parser.add_argument("--num_val_samples", default=1000, type=int)
    parser.add_argument("--min_num_iterations", "--min_iters", type=int, default=-1)
    parser.add_argument("--ignore_eval", action="store_true", default=False)
    parser.add_argument("--val_split", default="", type=str)
    parser.add_argument("--val_criterion", default="", type=str, choices=["", "loss", "accuracy"])
    parser.add_argument("--val_best_k_models", default=3, type=int, help="Num. best val models to keep while training")
    parser.add_argument("--val_patience", "--val_num_iters", "--val_num_iters_to_stop_after", default=0, type=int)
    parser.add_argument("--val_ratio", "--val_split_ratio", type=float, default=0.1, help="Ratio of the train set to separate as validation set (if applicable)")
    parser.add_argument("--val_keep_best_model", "--val_best", action="store_true", default=False)
    parser.add_argument("--evaluate_abtrain", action="store_true", default=False)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--so_seed", type=int, default=0, help="Subspace optimization (slicing) initialization seed.")
    parser.add_argument("--htm_gamma", type=float, default=0.0)
    parser.add_argument("--dampening", type=float, default=0.0)
    parser.add_argument("--layer_rank", "--rk", type=int, default=-1)
    parser.add_argument("--data_augmentation", "--aug", type=str, default="none")
    parser.add_argument("--matmul64", "--m64", action="store_true", default=False)
    parser.add_argument("--num_total_adv_samples", type=int, default=1000)
    parser.add_argument("--adv_samples_per_iter", type=int, default=100)
    parser.add_argument("--adv_sample_gen_freq", type=int, default=1000)
    parser.add_argument("--adv_repr_reg", "--arg", type=float, default=0.0)
    parser.add_argument("--adv_repr_reg_p", type=str, default="inf")
    parser.add_argument("--adv_training_norm", "--at_norm", type=str, default="")
    parser.add_argument("--adv_training_eps", "--at_eps", type=float, default=0.0)
    parser.add_argument("--adv_training_max_iters", "--at_iters", type=int, default=10)
    parser.add_argument("--adv_training_attacker", "--at_attacker", type=str, default="pgd")
    parser.add_argument("--adv_training_lr", "--at_lr", type=float, default=0.05882)
    parser.add_argument("--adv_training_ratio", "--at_ratio", type=float, default=0.5)



    args = parser.parse_args() #parsing arguments

    import os

    if args.cuda_device >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.cuda_device}'
    os.environ["OMP_NUM_THREADS"] = str(args.num_cpus) # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.num_cpus) # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = str(args.num_cpus) # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.num_cpus) # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.num_cpus) # export NUMEXPR_NUM_THREADS=1
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import datetime
    import time
    import copy
    import torch
    import torch.nn.functional as F
    if args.matmul64:
        torch.set_float32_matmul_precision('highest')
    else:
        torch.set_float32_matmul_precision('high')
    import sys
    import shutil
    from scipy.stats import levy_stable
    from models import get_model, get_grad_norms
    from utils import get_data, accuracy, update_avg_net, get_convergence_criteria, get_accuracy_lower_bound, write_message_to_file, write_experiment_script, cycle_loader, get_optimizer, get_criterion, evaluate_iteration, get_iterate_evaluation_msg, get_timestamp, get_scheduler, backup_src_files, npize, STATS, generate_x_adv
    import torch.nn as nn
    # if args.data_augmentation_plus:
    #     torch._dynamo.config.cache_size_limit = 512

    if args.float64:
        torch.set_default_dtype(torch.float64)
    if args.lr_scheduler == "none":
        args.lr_scheduler = ""
    torch.manual_seed(args.seed) # setting seed
    torch.use_deterministic_algorithms(True, warn_only=True)

    if not args.batch_size_eval:
        args.batch_size_eval = args.batch_size_train

    if args.double:
        torch.set_default_tensor_type('torch.DoubleTensor')
    loss_crit, accuracy_crit = get_convergence_criteria(args.dataset, args.convergence_method, args.convergence_loss, args.convergence_accuracy)

    args.device = torch.device(f'cuda' if args.cuda_device >= 0 else 'cpu')
    print(args.device)
    begin_time = time.time() #timer begun

    # Continuing from a previous iter.
    if not args.ignore_existing and os.path.exists(args.save_dir + "/training_iters") and not os.path.exists(args.save_dir + "/net.pyT"):
        iter_names = os.listdir(args.save_dir + "/training_iters")
        iters = sorted([int(it.split(".")[0].split("_")[1]) for it in iter_names])
        try:
            torch.load(args.save_dir + f"/training_iters/net_{iters[-1]}.pyT")
            max_iter = iters[-1]
        except:
            max_iter = iters[-2]
        shutil.copyfile(args.save_dir + f"/training_iters/net_{max_iter}.pyT", args.save_dir + "/net.pyT")
        eval_hist = torch.load(args.save_dir + "/evaluation_history.hist")
        train_hist = torch.load(args.save_dir + "/training_history.hist")
        torch.save(eval_hist, args.save_dir + "/evaluation_history_backup.hist")
        torch.save(train_hist, args.save_dir + "/training_history_backup.hist")
        eval_hist = [ev for ev in eval_hist if ev["iteration"] <= max_iter]
        train_hist = [tr for tr in train_hist if tr["iteration"] <= max_iter]
        torch.save(eval_hist, args.save_dir + "/evaluation_history.hist")
        torch.save(train_hist, args.save_dir + "/training_history.hist")

    # Saving script, if exists
    file_traj = args.save_dir + '_traj.log'

    script = "".join(args.script)
    args.script = ""
    write_message_to_file(f"PID: {os.getpid()}", file_traj, print_std=True)
    write_experiment_script(script, file_traj, print_std=True)
    # registering experiment

    write_message_to_file(str(args) + "\n", file_traj, print_std=True)
    os.makedirs(args.save_dir)
    torch.save(args, args.save_dir + "/args.info")

    args.device = torch.device(f'cuda' if args.cuda_device >= 0 else 'cpu')

    train_loader, eval_loaders, num_classes, input_dim = get_data(args, return_biased=True, abtrain=args.evaluate_abtrain, val_set=args.val_criterion) #done
    args.num_classes = num_classes

    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        net = get_model(model=args.model, device=args.device, input_dim=input_dim, width=args.width,
                    depth=args.depth, num_classes=num_classes, activation=args.activation, bias=args.use_bias,
                    custom_init=args.custom_init, img_channels=1 if (args.dataset == "mnist") or ("doublemnist" in args.dataset) else 3, layer_rank=args.layer_rank)
        # net = torch.compile(net)
    
    nets = {"net": net}

    write_message_to_file(str(net) + "\n", file_traj, print_std=True)

    opt = get_optimizer(net, args)
    crit = get_criterion(args)

    write_message_to_file(str(opt) + "\n", file_traj, print_std=True) if len(opt.param_groups) < 3 else write_message_to_file("Optimizer too long to print out.\n", file_traj, print_std=True)
    write_message_to_file(str(crit) + "\n", file_traj, print_std=True)
    if args.lr_scheduler:
        scheduler = get_scheduler(args.lr_scheduler, optimizer=opt, milestones=args.lrs_milestones, gamma=args.lrs_gamma)
    ### EVAL BEFORE FIRST STEP ### 
    if args.check_batch_convergence:
        eval_time = time.time()

        with torch.random.fork_rng():
            torch.manual_seed(args.seed)
            iterate_evals, iterate_outputs = evaluate_iteration(nets, eval_loaders, 999, crit, args, print_std=False, batch_convergence=False, ignore_eval=args.ignore_eval)
        write_message_to_file(f"Light evaluation taking {np.round(time.time() - eval_time, 3)} s.\n", file_traj, print_std=True)
    eval_time = time.time()

    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        iterate_evals, iterate_outputs = evaluate_iteration(nets, eval_loaders, 0, crit, args, print_std=False, ignore_eval=args.ignore_eval)
    write_message_to_file(f"Full evaluation taking {np.round(time.time() - eval_time, 3)} s.\n", file_traj, print_std=True)
    write_message_to_file(get_iterate_evaluation_msg(iterate_evals, eval_loaders.keys()), file_traj, print_std=True)

    training_history = [{"iteration": 0, "loss": np.nan, "accuracy": np.nan, "grad_norm": np.array([np.nan for p in net.parameters()]), "par_norm": torch.stack([p.data.norm() for p in net.parameters()]).cpu().numpy(), "lr": args.lr}]
    if "bias_conf_ratio" in args.track_statistics:
        training_history[-1]["bias_conf_ratio"] = 0.0
    evaluation_history = [iterate_evals,]

    prev_iters = 0    
    if script:
        write_message_to_file(script, args.save_dir + '/script.sh')
    if args.val_criterion:
        os.mkdir(args.save_dir + "/val_iters")
        val_best_k_models = []



    write_message_to_file("Backing up source files.\n", file_traj, print_std=True) 
    circ_train_loader = cycle_loader(train_loader)

    convergence_first_observed_at = 0

    if args.iter_record_freq != 0:
        training_iters_folder = args.save_dir + "/" + "training_iters/"
        if not os.path.exists(training_iters_folder):
            os.makedirs(training_iters_folder)
        if (args.iter_record_freq == 1) or (args.iter_record_freq < 0):
            torch.save(net, training_iters_folder + f'net_{0}.pyT')


    STOP = False

    iter_record_freq = args.iter_record_freq
    eval_freq = args.eval_freq
    torch.manual_seed(args.seed) 
    train_time = time.time()
    num_epochs = 0
    iters_per_epoch = np.ceil(len(train_loader.dataset)/args.batch_size_train)
    for j, (idx, x, y, *b) in enumerate(circ_train_loader):   
        i = j + 1
        if args.ignore_existing_data_cycle:
            i += prev_iters
        if i <= prev_iters:
            continue

        x, y = x.to(args.device), y.to(args.device)

        if args.adv_training_norm:
            net.eval()
            with torch.random.fork_rng():
                torch.manual_seed(args.seed + i)
                adv_idx = torch.randperm(len(x))[:int(args.adv_training_ratio * len(x))]
            stats = STATS[args.dataset]
            x_adv, _ = generate_x_adv(
                net, npize(x[adv_idx].clone()), args.adv_training_norm, x[0].shape, num_classes,
                stats, crit, args.device, attack_eps=args.adv_training_eps, attack_lr=args.adv_training_lr, 
                attacks_max_iter=args.adv_training_max_iters, seed=args.seed + i, attacker=args.adv_training_attacker, evaluation=False
                )
            x[adv_idx] = x_adv
        net.train()
        if b:
            b = b[0].to(args.device)
        opt.zero_grad()
        if "sam" in args.optimizer:
            enable_running_stats(net)

        out = net(x)

        loss = crit(out, F.one_hot(y, num_classes=num_classes).float() if args.criterion.upper() == 'MSE' else y)

        if args.l1 > 0.0:
            # print("L1!")
            loss_l1 = 0
            for parm in net.parameters():
                loss_l1 += torch.sum(torch.abs(parm))
            loss += args.l1 * loss_l1
    
        if args.rowwise_l2:
            assert args.layer_rank == -1
            first_conv = True
            loss_rowwise_l2 = 0
            for name, layer in net.named_modules():
                if not list(layer.parameters(recurse=False)):
                    continue # not a parameter
                if not hasattr(layer, "weight"):
                    continue # not a parameter we're interested in
                if isinstance(layer, nn.Linear):
                    if (layer.weight.shape[0] == num_classes):
                        continue  # do not regularize classifier head
                    if args.scale_invariant_row2ln:
                        loss_rowwise_l2 += args.rowwise_l2 * (layer.weight.norm(dim=1, p=2).sum()/layer.weight.norm(dim=1, p=2).norm(p=2))
                    else:
                        loss_rowwise_l2 += args.rowwise_l2 * layer.weight.norm(dim=1, p=2).sum()
                if isinstance(layer, nn.Conv2d): # Do not regularize the input convolution.
                    if first_conv:
                        first_conv = False
                        continue
                    if args.scale_invariant_row2ln:
                        loss_rowwise_l2 += args.rowwise_l2 * (layer.weight.view(layer.weight.shape[0], -1).norm(dim=1, p=2).sum()/layer.weight.view(layer.weight.shape[0], -1).norm(dim=1, p=2).norm(p=2))
                        
                    else:
                        loss_rowwise_l2 += args.rowwise_l2 * layer.weight.view(layer.weight.shape[0], -1).norm(dim=1, p=2).sum()
            loss += loss_rowwise_l2
        
        if args.nuclear_norm:
            nuclear_norm = 0
            total_spectral_variance = 0
            # HACK: Assuming no bias parameters
            parameters = list(net.parameters())
            for parm in parameters[:-1]:
                if parm.dim() == 2:
                    singular_values = torch.linalg.svdvals(parm)
                    nuclear_norm += args.nuclear_norm * torch.sum(singular_values)
                
            loss += nuclear_norm


        loss.backward()
    
        # record training history (starts at initial point)
        if (i % args.train_stats_freq == 0) or (i % eval_freq == 0) or (i == prev_iters):
            # print(net.fc[0].weight.data)
            training_history.append({
                "iteration": i,
                "loss": loss.item(),
                "accuracy": accuracy(out, y).item(),
                "grad_norm": torch.stack(get_grad_norms(net)).cpu().numpy() if not args.linear_head else np.nan,
                "par_norm": torch.stack([p.data.norm() for p in net.parameters()]).cpu().numpy(),
                "lr": scheduler.get_last_lr() if args.lr_scheduler else args.lr
                })        
            if "bias_conf_ratio" in args.track_statistics:
                training_history[-1]["bias_conf_ratio"] = (y.cpu().numpy() != b.cpu().numpy()).mean()
        if args.alpha > 0:
            for group in opt.param_groups:
                gan = (args.lr / args.width) ** (1 / (1 - args.alpha))
                group['lr'] = args.lr * (i + (1 / gan)) ** (- args.alpha)
        # take the step
        if "batch_stats_really" in args.track_statistics:
                batch_stats = {}
                batch_stats["logits"] = out.detach().cpu().numpy()
                batch_stats["idx"] = idx.cpu().numpy()
                batch_stats["y"] = y.cpu().numpy()
                batch_stats["b"] = b.cpu().numpy() if len(b) else np.array([np.nan for i in y])
                torch.save(batch_stats, args.save_dir + f"/bs_{i}.pyT")
        
        opt.step()
    
        if (iter_record_freq > 0) and (i % iter_record_freq == 0):
            torch.save(net,     training_iters_folder +     f'net_{i}.pyT')
            if args.traj:
                torch.save(avg_net, training_iters_folder + f'avg_net_{i}.pyT')
        if (i % eval_freq == 0) or (i == prev_iters):
            if i // eval_freq == 1:
                write_message_to_file(f"Approximate time per iteration: {np.round((time.time() - train_time)/i, 3)} s.\n", file_traj, print_std=True)
    
            train_b_eval = {"loss": np.nan, "accuracy": np.nan}
            with torch.random.fork_rng():
                torch.manual_seed(args.seed)
                iterate_evals, iterate_outputs = evaluate_iteration(nets, eval_loaders, i, crit, args, print_std=False, batch_convergence=False, ignore_eval=args.ignore_eval)
            splits = list(eval_loaders.keys())
            
            evaluation_history.append(iterate_evals)
            write_message_to_file(get_iterate_evaluation_msg(iterate_evals, splits, train_b_eval), file_traj, print_std=True)
            # print(f"Eval {np.round(time.time()-time_eval, 3)} s.\n")
            torch.save(training_history, args.save_dir + '/training_history.hist')
            torch.save(evaluation_history, args.save_dir + '/evaluation_history.hist')
            if os.path.exists(args.save_dir + "/STOP"):
                write_message_to_file("Cancelling the experiment manually.", file_traj, print_std=True)
                raise Exception("Experiment cancelled.")
            if os.path.exists(args.save_dir + "/PAUSE"):
                write_message_to_file("Pausing the experiment manually -- the model will be saved.", file_traj, print_std=True)
                STOP = True
            if args.lr_scheduler:
                write_message_to_file(f"LR: {[np.round(lr, 6) for lr in scheduler.get_last_lr()]}\n", file_traj, print_std=True)
            if args.val_criterion:
                coef = -1 if args.val_criterion == "accuracy" else 1
                val_var = f"eval_{args.val_split}_{args.val_criterion}"
                if (i >= args.min_num_iterations) and (len(val_best_k_models) < args.val_best_k_models):
                    val_best_k_models.append(evaluation_history[-1])
                    torch.save(net, args.save_dir + f"/val_iters/net_{i}.pyT")
                    val_best_k_models = [val_best_k_models[val_idx] for val_idx in np.argsort([coef * (m[val_var]) for m in val_best_k_models])]
                    write_message_to_file(f"Added model at iter {i} w/ {args.val_criterion} {evaluation_history[-1][val_var]} to best k models.\n", file_traj, print_std=True)
                else:
                    if (i >= args.min_num_iterations) and (coef * evaluation_history[-1][val_var]) < (coef * val_best_k_models[-1][val_var]):
                        os.remove(args.save_dir + f"/val_iters/net_{val_best_k_models[-1]['iteration']}.pyT")
                        torch.save(net, args.save_dir + f"/val_iters/net_{i}.pyT")
                        write_message_to_file(f"Added model at iter {i} w/ {args.val_criterion} {evaluation_history[-1][val_var]} to best k models (removed {val_best_k_models[-1]['iteration']} w/ {val_best_k_models[-1][val_var]}).\n", file_traj, print_std=True)
                        val_best_k_models.append(evaluation_history[-1])
                        val_best_k_models = [val_best_k_models[val_idx] for val_idx in np.argsort([coef * (m[val_var]) for m in val_best_k_models])[:args.val_best_k_models]]
                    torch.save(val_best_k_models, args.save_dir + f"/val_history.hist")
                if args.val_patience and (i >= args.min_num_iterations) and ((i - np.array([m["iteration"] for m in val_best_k_models]).max()) >= args.val_patience):
                    write_message_to_file(f"Training finished due to no improvement in {args.val_criterion} on the {args.val_split} split for {args.val_patience} iterations.", file_traj, print_std=True)
                    STOP = True
                    
        # update lr if a scheduler exists and its an epoch change
        if (j > 1) and not (j % iters_per_epoch):
            num_epochs += 1
            print(f"Epoch {num_epochs}")
            if args.lr_scheduler:
                scheduler.step()
                print(f"Scheduler LR updated.")

        if i == args.iterations: # changed this to equality
            STOP = True

        if np.isnan(loss.item()):
            write_message_to_file('Training terminated due to divergence.\n', file_traj, print_std=True)
            print(out)
            sys.exit()

        if (i >= args.num_improvement_iters) and (iterate_evals["eval_train_accuracy"] <= get_accuracy_lower_bound(num_classes)) and (args.eval_freq < 100000):
            write_message_to_file('Training terminated due to no increase in accuracy.\n', file_traj, print_std=True)
            sys.exit()
        
        

        if ((iterate_evals["eval_train_loss"] < loss_crit) and (iterate_evals["eval_train_accuracy"] >= accuracy_crit) and (i >= args.min_num_iterations)):
            print(f"Criteria met at {i}")
            print(args.mc_iterations)
            if convergence_first_observed_at == 0:
                convergence_first_observed_at = i
                burn_in_iterations = i + (mc_convergence_coefficient-1) * args.mc_iterations
            if (i - convergence_first_observed_at >= mc_convergence_coefficient * args.mc_iterations):
                write_message_to_file(f'Finishing training due to convergence, with loss < {loss_crit} and accuracy > {accuracy_crit}, (additional {mc_convergence_coefficient * args.mc_iterations} iterations after convergence).\n', file_traj, print_std=True)
                STOP = True
                


        

        if STOP:
            # Now a final evaluation is redundant unless %eval_freq or validation reload:
            if args.val_keep_best_model: 
                torch.save(net, args.save_dir + "/val_iters/net_final.pyT")
                write_message_to_file(f'Keeping the best validation model at Iter {val_best_k_models[0]["iteration"]}. Saved final model at {i} to val_iters/net_final.pyT instead.\n', file_traj, print_std=True)
                i = val_best_k_models[0]["iteration"]
                net = torch.load(args.save_dir + f"/val_iters/net_{i}.pyT")
                nets = {"net": net}
            if (evaluation_history[-1]["iteration"] != i) or (args.ignore_eval) or (args.val_criterion and args.val_keep_best_model):
                if args.ignore_eval and not (evaluation_history[-1]["iteration"] != i):
                    del evaluation_history[-1]
                    del training_history[-1]
                with torch.random.fork_rng():
                    torch.manual_seed(args.seed)
                    iterate_evals, iterate_outputs = evaluate_iteration(nets, eval_loaders, i, crit, args, print_std=False, ignore_eval=False)
                evaluation_history.append(iterate_evals)
                training_history.append({
                    "iteration": i,
                    "loss": loss.item() if not args.val_keep_best_model else np.nan,
                    "accuracy": accuracy(out, y).item() if not args.val_keep_best_model else np.nan,
                    "grad_norm": torch.stack(get_grad_norms(net)).cpu().numpy() if not args.linear_head and not args.val_keep_best_model else np.nan,
                    "par_norm": torch.stack([p.data.norm() for p in net.parameters()]).cpu().numpy(),
                    "lr": scheduler.get_last_lr() if args.lr_scheduler else args.lr
                    })
                write_message_to_file(get_iterate_evaluation_msg(iterate_evals, eval_loaders.keys()), file_traj, print_std=True)
                torch.save(training_history, args.save_dir + '/training_history.hist')
                torch.save(evaluation_history, args.save_dir + '/evaluation_history.hist')

            # save the setup
            torch.save(args, args.save_dir + '/args.info')
            # save the outputs
            torch.save(iterate_outputs["eval_train"], args.save_dir + '/tr_outputs.pyT')
            torch.save(iterate_outputs["eval_test"], args.save_dir + '/te_outputs.pyT')
            # if exists, save the script
            if args.traj:
                torch.save(iterate_outputs["avg_eval_train"], args.save_dir + '/ta_outputs.pyT')
                torch.save(iterate_outputs["avg_eval_test"], args.save_dir + '/tat_outputs.pyT')
                torch.save(avg_net, args.save_dir + '/avg_net.pyT')

            # save the model
            torch.save(net, args.save_dir + '/net.pyT')

            if args.iter_record_freq != 0:
                torch.save(net,     training_iters_folder +     f'net_{i}.pyT')
                if args.traj:
                    torch.save(avg_net, training_iters_folder + f'avg_net_{i}.pyT')

            torch.save(training_history, args.save_dir + '/training_history.hist')
            torch.save(evaluation_history, args.save_dir + '/evaluation_history.hist')

            end_time = time.time()
            total_time = end_time - begin_time
            time_secs = str(datetime.timedelta(seconds=total_time))

            write_message_to_file(f'Training finished. Total time: {time_secs}\n\n', file_traj, print_std=True)
            break

if __name__ == '__main__':
    main()