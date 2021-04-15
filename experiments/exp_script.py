import math
import numpy as np
import torch
import gpytorch
import tqdm
import argparse
import random
import time
import sys
import pickle
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
sys.path.append("../")
sys.path.append("../directionalvi/utils")
sys.path.append("../directionalvi")
from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad
from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy
from directional_vi import *
import grad_svgp
import traditional_vi
from load_data import *
from metrics import *
from synthetic_functions import *
from synthetic1.compute_optimal_subspace import *
try: # import wandb if watch model on weights&biases
  import wandb
except:
  pass

def main(**args):
    torch.set_default_dtype(torch.float64)
    torch.random.manual_seed(args["seed"])
    dataset_type = args["dataset"].split('-')[0]
    dataset_name = args["dataset"].split('-')[1]
    n = args["n_train"]
    n_test = args["n_test"]
    num_inducing = args["num_inducing"]
    num_directions = args["num_directions"]
    minibatch_size = args["batch_size"]
    num_epochs = args["num_epochs"]
    variational_dist = args["variational_distribution"]
    variational_strat = args["variational_strategy"]
    use_ngd=True if variational_dist == "NGD" else False
    use_ciq=True if variational_strat == "CIQ" else False
    learning_rate_hypers = args["lr"]
    learning_rate_ngd = args["lr_ngd"]
    num_contour_quadrature=args["num_contour_quad"]
    mll_type=args["mll_type"]
    lr_sched=args["lr_sched"]
    if lr_sched == "lambda_lr":
        lr_sched = lambda epoch: 1.0/(1 + epoch)
    elif lr_sched == "None":
        lr_sched = None
        
    exp_name = args["exp_name"]
    if args["model"]=="SVGP":
        args["derivative"]=False
        expname_train = f"{dataset_name}_{args['model']}_ntrain{n}_m{num_inducing}_epochs{num_epochs}_{variational_dist}_{variational_strat}_exp{exp_name}"
    elif args["model"]=="DSVGP" or args["model"]=="GradSVGP":
        args["derivative"]=True
        expname_train = f"{dataset_name}_{args['model']}_ntrain{n}_m{num_inducing}_p{num_directions}_epochs{num_epochs}_{variational_dist}_{variational_strat}_exp{exp_name}"
    expname_test = f"{expname_train}_ntest{n_test}"

    if args["watch_model"]: # watch model on weights&biases
        wandb.init(project='DSVGP', entity='xinranzhu',
                name=expname_test)
        print("Experiment settings:")
        print(args)
        # save hyperparameters with wandb
        wandb.config.learning_rate_hypers = learning_rate_hypers
        wandb.config.learning_rate_ngd = learning_rate_ngd
        wandb.config.lr_sched = lr_sched
        wandb.config.num_contour_quadrature = num_contour_quadrature
        wandb.config.mll_type = mll_type

    print(f"\n\n\nStart Experiment: {expname_test}")

    # generate training data, x in unit cube, y normalized, derivative rescaled
    if dataset_type=="synthetic":
        testfun = eval(f"{dataset_name}_with_deriv")()
        dim = testfun.dim
        x, y, info_dict = load_synthetic_data(testfun, n+n_test, **args)
        train_x = x[:n, :]
        test_x = x[n:, :]
        train_y = y[:n, ...]
        test_y = y[n:, ...]
        #obtain train and test TensorDatasets
        if torch.cuda.is_available():
            train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
    else: #load real dataset
        #obtain train and test TensorDatasets
        data_loader = eval(f"load_{dataset_name}")
        data_src_path = f"../data/{dataset_name}"
        args["filter_val"] = 1.0
        train_x, train_y, test_x, test_y, dim, info_dictt = data_loader(data_src_path, **args)
        n = info_dict["n_train"]
        n_test = info_dict["n_test"]
        
    assert num_inducing < n
    assert num_directions <= dim
    assert minibatch_size <= n

    # active subspace for GradSVGP
    if args["model"]=="GradSVGP" and num_directions < dim:
        G_train, train_x, P = compute_optimal_subspace_projection((train_y[:, 1:]).cpu().numpy(),train_x.cpu().numpy(),num_directions)
        P = torch.tensor(P, dtype=train_y.dtype)
        G_train = torch.tensor(G_train, dtype=train_y.dtype)
        train_x = torch.tensor(train_x, dtype=test_x.dtype)
        if torch.cuda.is_available():
            P, G_train,train_x = P.cuda(), G_train.cuda(), train_x.cuda(), 
        train_y = torch.cat([train_y[:,0:1],G_train], 1)
        test_x = test_x@P
        G_test = test_y[:,1:]@P
        test_y =  torch.cat([test_y[:,0:1], G_test], 1)

    train_dataset = TensorDataset(train_x,train_y)
    test_dataset = TensorDataset(test_x,test_y)

    # train
    print(f"\n---{args['model']}---")
    print(f"Variational distribution: {variational_dist}, Variational strategy: {variational_strat}")
    print(f"Start training with {n} trainig data of dim {dim}")
    print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        t1 = time.time_ns()	

    if args["model"]=="SVGP":
        model,likelihood = traditional_vi.train_gp(train_dataset,dim,
                                                   num_inducing=num_inducing,
                                                   minibatch_size=minibatch_size,
                                                   num_epochs=num_epochs,
                                                   use_ngd=use_ngd, use_ciq=use_ciq,
                                                   learning_rate_hypers=learning_rate_hypers,
                                                   learning_rate_ngd=learning_rate_ngd,
                                                   lr_sched=lr_sched,mll_type=mll_type,
                                                   num_contour_quadrature=num_contour_quadrature,
                                                   watch_model=args["watch_model"],gamma=args["gamma"])
    elif args["model"]=="DSVGP":
        model,likelihood = train_gp(train_dataset,
                                num_inducing=num_inducing,
                                num_directions=num_directions,
                                minibatch_size=minibatch_size,
                                minibatch_dim=num_directions,
                                num_epochs=num_epochs,
                                use_ngd=use_ngd, use_ciq=use_ciq,
                                lr_sched = lr_sched,mll_type=mll_type,
                                learning_rate_ngd = learning_rate_ngd,
                                learning_rate_hypers = learning_rate_hypers,
                                num_contour_quadrature = num_contour_quadrature,
                                watch_model=args["watch_model"],gamma=args["gamma"])
    elif args["model"]=="GradSVGP":
        model,likelihood = grad_svgp.train_gp(train_dataset,num_directions,num_inducing=num_inducing,
                                            minibatch_size=minibatch_size,
                                            num_epochs=num_epochs,
                                            use_ngd=use_ngd,use_ciq=use_ciq,
                                            learning_rate_hypers=learning_rate_hypers,
                                            learning_rate_ngd=learning_rate_ngd,
                                            lr_sched=lr_sched, mll_type=mll_type,
                                            num_contour_quadrature = num_contour_quadrature,
                                            watch_model=args["watch_model"],gamma=args["gamma"])

    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        train_time = start.elapsed_time(end)/1e3
        sys.stdout.flush()
        start.record()
    else:
        t2 = time.time_ns()
        train_time = (t2-t1)/1e9
        	

    # save the model
    if args["save_model"]:
        torch.save(model.state_dict(), f"../data/{expname_train}.model")


    # test
    if args["model"]=="SVGP":
        means, variances = traditional_vi.eval_gp(test_dataset,model,likelihood, 
                                                  num_inducing=num_inducing,
                                                  minibatch_size=minibatch_size)
        pred_means = means
        pred_variance = variances
        test_f = test_y
    elif args["model"]=="DSVGP":
        means, variances = eval_gp(test_dataset,model,likelihood,
                                    num_directions=num_directions,
                                    minibatch_size=minibatch_size,
                                    minibatch_dim=num_directions)
        pred_means = means[::num_directions+1]
        pred_variance = variances[::num_directions+1]
        test_f = test_y[:,0]
    elif args["model"]=="GradSVGP":
        means, variances = grad_svgp.eval_gp(test_dataset,model,likelihood,
                                            num_inducing=num_inducing,
                                            minibatch_size=minibatch_size)
        pred_means = means[::num_directions+1]
        pred_variance = variances[::num_directions+1]
        test_f = test_y[:,0]
    
    # metrics
    test_mse = MSE(test_f.cpu(),pred_means)
    test_rmse = RMSE(test_f.cpu(),pred_means)
    test_mae = MAE(test_f.cpu(),pred_means)
    test_smae = SMAE(test_f.cpu(),pred_means)
    test_nll = -torch.distributions.Normal(pred_means, pred_variance.sqrt()).log_prob(test_f.cpu()).mean()

    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        test_time = start.elapsed_time(end)/1e3
        sys.stdout.flush()
    else:    
        t3 = time.time_ns()
        test_time = (t3-t2)/1e9	
    print(f"At {n_test} testing points, RMSE: {test_rmse:.4e}, nll: {test_nll:.4e}, MAE: {test_mae:.4e}, MSE: {test_mse:.4e}, SMAE: {test_smae:.4e}")
    print(f"Training time: {train_time:.2f} sec, testing time: {test_time:.2f} sec")

    # save data for plots
    if args["save_results"]:
        summary=[n, train_dataset, n_test, test_dataset, means, info_dict]
        pickle.dump(summary,open(f"./postprocess/exp_res/{expname_test}.pkl","wb"))


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="parse args")

    # Directories for data/logs
    parser.add_argument("-ld", "--log-dir", type=str, default="./logs/")
    parser.add_argument("-dd", "--data-dir", type=str, default="./data/")
    parser.add_argument("-sm", "--save_model", type=str2bool, nargs='?',const=True, default=False)
    parser.add_argument("--watch_model", type=str2bool, nargs='?',const=True, default=False) 
    parser.add_argument("--save_results", type=str2bool,nargs='?',const=True, default=False) #exp_script.py: error: argument --save_results: expected one argument
    parser.add_argument("--exp_name", type=str, default="-")

    # Dataset and model type
    #TODO: add real dataset experiment
    parser.add_argument("-d", "--dataset", type=str, default="synthetic-Branin")
    parser.add_argument("--model", type=str, default="DSVGP")
    parser.add_argument("-vs", "--variational_strategy", type=str, default="standard", choices=["standard", "CIQ"])
    parser.add_argument("-vd", "--variational_distribution", type=str, default="standard", choices=["standard", "NGD"])

    # Model args
    # Training args
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("-m", "--num_inducing", type=int, default=10)
    parser.add_argument("-p", "--num_directions", type=int, default=10)
    parser.add_argument("-n", "--num_epochs", type=int, default=1)
    parser.add_argument("-bs", "--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_ngd", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--num_contour_quad", type=int, default=15)
    parser.add_argument("--lr_sched", type=str, default=None)
    parser.add_argument("--mll_type", type=str, default="ELBO", choices=["ELBO", "PLL"])
    # Seed/splits/restarts
    parser.add_argument("-s", "--seed", type=int, default=0)
    # parser.add_argument("-ns", "--num-splits", type=int, default=1)
    # parser.add_argument("-nr", "--num-restarts", type=int, default=0)

    args = parser.parse_args()
    main(**vars(args))

