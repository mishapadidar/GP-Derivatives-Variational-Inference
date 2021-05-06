import pymesh
from scipy import io
import numpy as np
from sklearn import preprocessing
import random
import math
import torch
import gpytorch
import tqdm
import argparse
import time
import sys
import pickle
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
sys.path.append("../../")
sys.path.append("../../directionalvi/utils")
sys.path.append("../../directionalvi")
from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad
from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy
from directional_vi import *
import grad_svgp
import traditional_vi
from load_data import *
from metrics import *
from synthetic_functions import *
from scipy.io import savemat
import wandb

def main(**args):
    dim = 3

    mat = io.loadmat("./ImplicitBunny/bunny.mat")
    X = mat['obj']['v'][0][0]
    Xorig = X
    T = mat['obj']['f'][0][0][0][0][0]
    Torig = T
    nx = mat['obj']['vn'][0][0]
    nx = preprocessing.normalize(nx, norm='l2')

    def mapToUnitbox(X):
        minX = np.min(X,0)
        lenX = np.max(X,0)- minX
        X = (X - minX)/lenX
        return X
    X2 = mapToUnitbox(X)
    lims = [-0.01, 1.01,     -0.01, 1.01,     -0.01, 1.01]


    # add noise 
    noise = 0.01
    X = X + noise*np.random.random((X.shape[0], dim))
    nx = nx + 0*np.random.random((X.shape[0], dim))

    # Map back to unitbox and pick subset
    X = mapToUnitbox(X)
    nn = 3; 
    x = X[::nn,0]
    y = X[::nn,1]
    z = X[::nn,2]
    nx_train = nx[::nn, :]
    n = x.shape[0]
    n_test = X.shape[0] - n
    print(f'Size of Kdot: [{ n*(dim+1)} { n*(dim+1)}]\n')

    torch.set_default_dtype(torch.float64)
    torch.random.manual_seed(args["seed"])
    dataset_type = args["dataset"].split('-')[0]
    dataset_name = args["dataset"].split('-')[1]
    num_inducing = args["num_inducing"]
    num_directions = args["num_directions"]
    minibatch_size = args["batch_size"]
    num_epochs = args["num_epochs"]
    variational_dist = args["variational_distribution"]
    variational_strat = args["variational_strategy"]
    exp_name = args["exp_name"]
    use_ngd=True if variational_dist == "NGD" else False
    use_ciq=True if variational_strat == "CIQ" else False
    learning_rate_hypers = args["lr"]
    learning_rate_ngd = args["lr_ngd"]
    num_contour_quadrature=args["num_contour_quad"]
    lr_sched=args["lr_sched"]
    mll_type=args["mll_type"]
    if lr_sched == "lambda_lr":
        lr_sched = lambda epoch: 1.0/(1 + epoch)
    elif lr_sched == "None":
        lr_sched = None
        
    if args["model"]=="SVGP":
        args["derivative"]=False
        expname_train = f"{dataset_name}_{args['model']}_ntrain{n}_m{num_inducing}_epochs{num_epochs}_{variational_dist}_{variational_strat}_exp{exp_name}"
    elif args["model"]=="DSVGP":
        args["derivative"]=True
        expname_train = f"{dataset_name}_{args['model']}_ntrain{n}_m{num_inducing}_p{num_directions}_epochs{num_epochs}_{variational_dist}_{variational_strat}_exp{exp_name}"
    elif args["model"]=="GradSVGP":
        args["derivative"]=True
        expname_train = f"{dataset_name}_{args['model']}_ntrain{n}_m{num_inducing}_epochs{num_epochs}_{variational_dist}_{variational_strat}_exp{exp_name}"
    expname_test = f"{expname_train}_ntest{n_test}"

    if args["watch_model"]:
        wandb.init(project='DSVGP', entity='xinranzhu', name=expname_test)
        print("Experiment settings:")
        print(args)
        # save hyperparameters with wandb
        wandb.config.learning_rate_hypers = learning_rate_hypers
        wandb.config.learning_rate_ngd = learning_rate_ngd
        wandb.config.lr_sched = lr_sched
        wandb.config.num_contour_quadrature = num_contour_quadrature
        wandb.config.mll_type = mll_type

    print(f"\n\n\nStart Experiment: {expname_test}")
    kwargs = {"seed":0, "derivative":True}


    train_x = torch.cat([torch.tensor(x).reshape(-1,1),
                        torch.tensor(y).reshape(-1,1),
                        torch.tensor(z).reshape(-1,1)],1)
    train_f = torch.zeros(train_x.shape[0], 1)
    train_y = torch.cat([train_f, torch.tensor(nx_train)],1)
    if torch.cuda.is_available():
        train_x, train_y, = train_x.cuda(), train_y.cuda()
            
    train_dataset = TensorDataset(train_x,train_y)


    # train
    print(f"\n---{args['model']}---")
    print(f"Variational distribution: {variational_dist}, Variational strategy: {variational_strat}")
    print(f"Start training with {n} trainig data of dim {dim}")
    print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")

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

    # save model 
    torch.save(model.state_dict(), f"./{expname_train}.model")

    # prediction
    isize = 100
    nxx = isize
    nyy = isize
    nzz = isize
    x1 = np.linspace(lims[0], lims[1], nxx)
    x2 = np.linspace(lims[2], lims[3], nyy) 
    x3 = np.linspace(lims[4], lims[5], nzz)

    [XX, YY, ZZ] = np.meshgrid(x1, x2, x3)
    X = np.concatenate([XX.reshape(-1, 1, 1).squeeze(-1), 
                    YY.reshape(-1, 1, 1).squeeze(-1), 
                    ZZ.reshape(-1, 1, 1).squeeze(-1)], 1)
    test_x = torch.tensor(X)
    test_y = torch.zeros((X.shape[0], dim+1))
    if torch.cuda.is_available():
        test_x, test_y = test_x.cuda(), test_y.cuda()
    test_dataset = TensorDataset(test_x,test_y)

    means, _ = eval_gp(test_dataset,model,likelihood,
                        num_directions=num_directions,
                        minibatch_size=minibatch_size,
                        mll_type=mll_type,
                        minibatch_dim=num_directions)

    pred_y = means.detach().cpu().numpy()[::num_directions+1]
    V = pred_y.reshape((nxx, nyy, nzz))
    res_dict = {"V": V, "means": means.detach().cpu().numpy()}
    savemat(f"./results/{expname_test}_res.mat", res_dict)




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
    parser.add_argument("-sm", "--save_model", type=str2bool, nargs='?',const=True, default=True)
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
    # parser.add_argument("--n_train", type=int, default=100)
    # parser.add_argument("--n_test", type=int, default=100)
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

