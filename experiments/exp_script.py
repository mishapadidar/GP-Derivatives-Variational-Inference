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
import traditional_vi
from load_data import *
from metrics import *
from synthetic_functions import *
try: # import wandb if watch model on weights&biases
  import wandb
except:
  pass

def main(**args):
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
    lr_sched=args["lr_sched"]
    if lr_sched == "lambda_lr":
        lr_sched = lambda epoch: 1.0/(1 + epoch)
    elif lr_sched == "None":
        lr_sched = None
        
    exp_name = args["exp_name"]
    if args["model"]=="SVGP":
        args["derivative"]=False
        expname_train = f"{dataset_name}_{args['model']}_ntrain{n}_m{num_inducing}_epochs{num_epochs}_{variational_dist}_{variational_strat}_exp{exp_name}"
    elif args["model"]=="DSVGP":
        args["derivative"]=True
        expname_train = f"{dataset_name}_{args['model']}_ntrain{n}_m{num_inducing}_p{num_directions}_epochs{num_epochs}_{variational_dist}_{variational_strat}_exp{exp_name}"
    expname_test = f"{expname_train}_ntest{n_test}"

    if args["watch_model"]: # watch model on weights&biases
        wandb.init(project='DSVGP', entity='jimmypotato',
                name=expname_test)
        print("Experiment settings:")
        print(args)
        # save hyperparameters with wandb
        wandb.config.learning_rate_hypers = learning_rate_hypers
        wandb.config.learning_rate_ngd = learning_rate_ngd
        wandb.config.lr_sched = lr_sched
        wandb.config.num_contour_quadrature = num_contour_quadrature

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
        train_dataset = TensorDataset(train_x,train_y)
        test_dataset = TensorDataset(test_x,test_y)
    else: #load real dataset
        #obtain train and test TensorDatasets
        data_loader = eval(f"load_{dataset_name}")
        data_src_path = f"../data/{dataset_name}"
        filter_val = 1.0
        train_dataset, test_dataset, dim, info_dict = data_loader(data_src_path, filter_val, **args)
        n = info_dict["n_train"]
        n_test = info_dict["n_test"]
        
    assert num_inducing < n
    assert num_directions <= dim
    assert minibatch_size <= n

    # train
    if args["model"]=="SVGP":
        print("\n---Traditional SVGP---")
        print(f"Variational distribution: {variational_dist}, Variational strategy: {variational_strat}")
        print(f"Start training with {n} trainig data of dim {dim}")
        print(f"VI setups: {num_inducing} inducing points")
    elif args["model"]=="DSVGP":
        print("\n---D-SVGP---")
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
                                                   use_ngd=use_ngd,
                                                   use_ciq=use_ciq,
                                                   learning_rate_hypers=learning_rate_hypers,
                                                   learning_rate_ngd=learning_rate_ngd,
                                                   lr_sched=lr_sched,
                                                   num_contour_quadrature=num_contour_quadrature,
                                                   tqdm=False,
                                                   watch_model=args["watch_model"])
    elif args["model"]=="DSVGP":
        model,likelihood = train_gp(train_dataset,
                                num_inducing=num_inducing,
                                num_directions=num_directions,
                                minibatch_size=minibatch_size,
                                minibatch_dim=num_directions,
                                num_epochs=num_epochs,
                                tqdm=False, use_ngd=use_ngd, use_ciq=use_ciq,
                                lr_sched = lr_sched,
                                learning_rate_ngd = learning_rate_ngd,
                                learning_rate_hypers = learning_rate_hypers,
                                num_contour_quadrature = num_contour_quadrature,
                                watch_model=args["watch_model"]
                                )

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

    # collect the test function values
    test_f = torch.zeros(n_test)
    for ii in range(n_test):
        if args["model"] == "DSVGP":
            test_f[ii] = test_dataset[ii][1][0] # function value
        elif args["model"] == "SVGP":
            test_f[ii] = test_dataset[ii][1] # function value

    # test
    if args["model"]=="SVGP":
        means, variances = traditional_vi.eval_gp(test_dataset,model,likelihood, 
                                                  num_inducing=num_inducing,
                                                  minibatch_size=minibatch_size)
        # metrics
        test_mse = MSE(test_f.cpu(),means)
        test_rmse = RMSE(test_f.cpu(),means)
        test_mae = MAE(test_f.cpu(),means)
        test_nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_f.cpu()).mean()
    elif args["model"]=="DSVGP":
        means, variances = eval_gp( test_dataset,model,likelihood,
                                    num_directions=num_directions,
                                    minibatch_size=minibatch_size,
                                    minibatch_dim=num_directions)
        # compute MSE
        test_mse = MSE(test_f.cpu(),means[::num_directions+1])
        test_rmse = RMSE(test_f.cpu(),means[::num_directions+1])
        test_mae = MAE(test_f.cpu(),means[::num_directions+1])
        # compute mean negative predictive density
        test_nll = -torch.distributions.Normal(means[::num_directions+1], variances.sqrt()[::num_directions+1]).log_prob(test_f.cpu()).mean()
    
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        test_time = start.elapsed_time(end)/1e3
        sys.stdout.flush()
    else:    
        t3 = time.time_ns()
        test_time = (t3-t2)/1e9	
    #print("hi")
    print(f"At {n_test} testing points, MSE: {test_mse:.4e}, RMSE: {test_rmse:.4e}, MAE: {test_mae:.4e}, nll: {test_nll:.4e}.")
    print(f"Training time: {train_time:.2f} sec, testing time: {test_time:.2f} sec")

    # save data for plots
    if args["save_results"]:
        summary=[n, train_dataset, n_test, test_dataset, means, info_dict]
        pickle.dump(summary,open(f"./postprocess/exp_res/{expname_test}.pkl","wb"))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="parse args")

    # Directories for data/logs
    parser.add_argument("-ld", "--log-dir", type=str, default="./logs/")
    parser.add_argument("-dd", "--data-dir", type=str, default="./data/")
    parser.add_argument("-sm", "--save_model", type=bool, default=False)
    parser.add_argument("--watch_model", type=bool, default=False) 
    parser.add_argument("--save_results", type=bool, default=False) #exp_script.py: error: argument --save_results: expected one argument
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
    parser.add_argument("--num_contour_quad", type=int, default=15)
    parser.add_argument("--lr_sched", type=str, default=None)

    # Seed/splits/restarts
    parser.add_argument("-s", "--seed", type=int, default=0)
    # parser.add_argument("-ns", "--num-splits", type=int, default=1)
    # parser.add_argument("-nr", "--num-restarts", type=int, default=0)

    args = parser.parse_args()
    main(**vars(args))

