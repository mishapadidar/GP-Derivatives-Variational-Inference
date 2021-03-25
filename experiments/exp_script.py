import math
import numpy as np
import torch
import gpytorch
import tqdm
import argparse
import random
import time
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.append("../")
sys.path.append("../directionalvi/utils")
sys.path.append("../directionalvi")
from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad
from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy
from directional_vi import train_gp, eval_gp
from metrics import MSE
from synthetic.test_funs import *

def main(**args):
    # seed
    torch.random.manual_seed(args["seed"])

    testfun_name = f"{args['test_fun']}"
    testfun = eval(f"{testfun_name}_with_deriv")()
    dim = testfun.dim
    n = args["n_train"]
    n_test = args["n_test"]
    num_inducing = args["num_inducing"]
    num_directions = args["num_directions"]
    minibatch_size = args["batch_size"]
    num_epochs = args["num_epochs"]
    variational_dist = args["variational_distribution"]
    variational_strat = args["variational_strategy"]
    assert num_inducing < n
    assert num_directions <= dim
    assert minibatch_size <= n
    expname_train = f"{testfun_name}_ntrain{n}_m{num_inducing}_p{num_directions}_epochs{num_epochs}_{variational_dist}_{variational_strat}_seed{args['seed']}"
    expname_test = f"{expname_train}_ntest{n_test}"
    print(f"\nStart Experiment: expname_test")

    # generate training data
    train_x = torch.rand(n,dim)
    train_y = testfun.evaluate_true_with_deriv(train_x)
    # testing data
    test_x = torch.rand(n_test,dim)
    test_y = testfun.evaluate_true_with_deriv(test_x)
    # if torch.cuda.is_available():
    #     train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
    train_dataset = TensorDataset(train_x,train_y)
    test_dataset = TensorDataset(test_x,test_y)

    # train
    print("\n\n---DirectionalGradVGP---")
    print(f"Start training with {n} trainig data of dim {dim}")
    print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")
    t1 = time.time()	
    if args["variational_distribution"] == "standard" and args["variational_strategy"] == "standard":
        model,likelihood = train_gp(train_dataset,
                            num_inducing=num_inducing,
                            num_directions=num_directions,
                            minibatch_size = minibatch_size,
                            minibatch_dim = num_directions,
                            num_epochs =num_epochs,
                            tqdm=False
                            )
    elif args["variational_distribution"] == "NGD" and args["variational_strategy"] == "standard":
        model,likelihood = train_gp_ngd(train_dataset,
                            num_inducing=num_inducing,
                            num_directions=num_directions,
                            minibatch_size = minibatch_size,
                            minibatch_dim = num_directions,
                            num_epochs =num_epochs,
                            tqdm=False
                            )
    elif args["variational_strategy"] == "CIQ":
        model,likelihood = train_gp_ciq(train_dataset,
                            num_inducing=num_inducing,
                            num_directions=num_directions,
                            minibatch_size = minibatch_size,
                            minibatch_dim = num_directions,
                            num_epochs =num_epochs,
                            tqdm=False
                            )

    t2 = time.time()	

    # save the model
    if args["save_model"]:
        torch.save(model.state_dict(), f"../data/{expname_train}.model")

    # test
    means, variances = eval_gp( test_dataset,model,likelihood,
                                num_inducing=num_inducing,
                                num_directions=num_directions,
                                minibatch_size=n_test,
                                minibatch_dim=num_directions,
                                num_epochs=1)
    t3 = time.time()	

    # compute MSE
    test_mse = MSE(test_y[:,0],means[::num_directions+1])
    # compute mean negative predictive density
    test_nll = -torch.distributions.Normal(means[::num_directions+1], variances.sqrt()[::num_directions+1]).log_prob(test_y[:,0]).mean()
    print(f"At {n_test} testing points, MSE: {test_mse:.4e}, nll: {test_nll:.4e}.")
    print(f"Training time: {(t2-t1)/1e9:.2f} sec, testing time: {(t3-t2)/1e9:.2f} sec")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="parse args")

    # Directories for data/logs
    parser.add_argument("-ld", "--log-dir", type=str, default="./logs/")
    parser.add_argument("-dd", "--data-dir", type=str, default="./data/")
    parser.add_argument("-sm", "--save_model", type=bool, default=False)

    # Dataset and model type
    #TODO: add real dataset experiment
    # parser.add_argument("-d", "--dataset", type=str, default="bunny")
    parser.add_argument("-f", "--test_fun", type=str, default="Branin")
    parser.add_argument("-vs", "--variational_strategy", type=str, default="standard", choices=["standard", "CIQ"])
    parser.add_argument("-vd", "--variational_distribution", type=str, default="standard", choices=["standard", "NGD"])

    # Model args
    #TODO: add CIQ args

    # Training args
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("-m", "--num_inducing", type=int, default=10)
    parser.add_argument("-p", "--num_directions", type=int, default=10)
    parser.add_argument("-n", "--num_epochs", type=int, default=1)
    parser.add_argument("-bs", "--batch_size", type=int, default=256)
    #TODO: add learning rates
    # parser.add_argument("-lr", "--lr", type=float, default=0.01)
    # parser.add_argument("-vlr", "--vlr", type=float, default=0.1)

    # Seed/splits/restarts
    parser.add_argument("-s", "--seed", type=int, default=0)
    # parser.add_argument("-ns", "--num-splits", type=int, default=1)
    # parser.add_argument("-nr", "--num-restarts", type=int, default=0)

    args = parser.parse_args()
    main(**vars(args))

