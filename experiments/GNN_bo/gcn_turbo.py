import math
import numpy as np
import random
import time
import argparse
import wandb

import torch
import gpytorch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
# import networkx as nx

from matplotlib import pyplot as plt

import os
import sys
sys.path.append("../")
sys.path.append("../../directionalvi/utils")
sys.path.append("../../directionalvi")
import directional_vi 
import traditional_vi
import shared_directional_vi
import grad_svgp
from metrics import MSE
import pickle
from scipy.io import loadmat
# from GCN.utils import *
# from GCN.models import GCN
from torch_geometric.datasets import Planetoid
from GCN.models2 import Net

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser(description="parse args")
# Directories for data/logs
# parser.add_argument("--watch_model", type=str2bool, nargs='?',const=True, default=False) 
parser.add_argument("--exp_name", type=str, default="-")
# Dataset and model type
parser.add_argument("-d", "--dataset", type=str, default="synthetic-Branin")
parser.add_argument("--model", type=str, default="DSVGP")
parser.add_argument("-vs", "--variational_strategy", type=str, default="standard", choices=["standard", "CIQ"])
parser.add_argument("-vd", "--variational_distribution", type=str, default="standard", choices=["standard", "NGD"])
parser.add_argument("-m", "--num_inducing", type=int, default=10)
parser.add_argument("-p", "--num_directions", type=int, default=10)
parser.add_argument("-n", "--num_epochs", type=int, default=1)
parser.add_argument("-bs", "--batch_size", type=int, default=256)
parser.add_argument("--turbo_batch_size", type=int, default=50)
parser.add_argument("--turbo_max_evals", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_ngd", type=float, default=0.1)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--num_contour_quad", type=int, default=15)
parser.add_argument("--lr_sched", type=str, default=None)
parser.add_argument("--mll_type", type=str, default="ELBO", choices=["ELBO", "PLL"])
parser.add_argument("-s", "--seed", type=int, default=0)


args =  vars(parser.parse_args())

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

turbo_max_evals = args["turbo_max_evals"]
turbo_batch_size = args["turbo_batch_size"]


exp_name = args["exp_name"]
if args["model"]=="SVGP":
  args["derivative"]=False
  expname_train = f"{args['dataset']}_{args['model']}_m{num_inducing}_epochs{num_epochs}_turboN{turbo_max_evals}_turbo_bs{turbo_batch_size}_exp{exp_name}"
elif args["model"]=="TURBO" or args["model"]=="BO":
  args["derivative"]=False
  expname_train = f"{args['dataset']}_{args['model']}_m{num_inducing}_epochs{num_epochs}_turboN{turbo_max_evals}_turbo_bs{turbo_batch_size}_exp{exp_name}"
elif args["model"]=="DSVGP" or args["model"]=="GradSVGP" or args["model"]=="DSVGP_shared":
  args["derivative"]=True
  expname_train = f"{args['dataset']}_{args['model']}_m{num_inducing}_p{num_directions}_epochs{num_epochs}_turboN{turbo_max_evals}_turbo_bs{turbo_batch_size}_exp{exp_name}"
elif args["model"]=="random":
  args["derivative"]=False
  expname_train = f"{args['dataset']}_{args['model']}_m{num_inducing}_turboN{turbo_max_evals}_exp{exp_name}"
expname_test = f"{expname_train}"
print(expname_test)
print("Args:\n")
print(args)

# output result file names
data_dir = "./results/"
model_filename = data_dir + expname_test + ".model"
data_filename  = data_dir + expname_test + ".pickle"
if os.path.exists(data_dir) is False:
  os.mkdir(data_dir)


torch.set_default_dtype(torch.float64)
torch.random.manual_seed(args["seed"])  

if torch.cuda.is_available():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
else:
    t1 = time.time_ns()	

# load data for GCN
dataset = "PubMed"
assert args["dataset"] == "PubMed"
dataset = Planetoid(root='/tmp/PubMed', name='PubMed')  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
turbo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(dataset).to(device)
data = dataset[0].to(device)
dim = model.n_params
print("\nDimension of GCN:", dim)

# initialize a GCN model
# model = GCN(nfeat=features.shape[1],
#             nhid=n_hidden_layers,
#             nclass=labels.max().item() + 1)


turbo_lb = np.repeat([-10.],dim)
turbo_ub = np.repeat([10.],dim)
turbo_n_init = 400 if num_inducing < 400 else num_inducing

# wrap the objective
def objective(w):
  model.train()
  # set the weights
  model.update_weights(torch.tensor(w, device=turbo_device))
  # predict
  output = model(data) 
  model.zero_grad()
  # compute the loss
  loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
  if args["derivative"]==True:
    # accumulate grad
    loss.backward()
    # get the grad
    grad = model.get_grad()
    # stack it
    if turbo_device == 'cpu':
      fg = np.zeros(len(w)+1)
    else:
      fg = torch.zeros(len(w)+1)
    fg[0] = loss.item()
    fg[1:] = grad
    # print("evaluating loss: ", loss.item())
    return fg
  else:
    # print("evaluating loss: ", loss.item())
    return loss.item()



if args["model"] == "DSVGP":
  # train
  print("\n\n---TuRBO-Grad with DSVGP in dim {dim}---")
  print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")

  gp_eval_batch_size = 1000
  def train_gp_for_turbo(train_x, train_y, num_steps):
    # expects train_x on unit cube and train_y standardized
    # make a trainable model for TuRBO
    train_x = train_x.float()
    train_y = train_y.float()
    dataset = TensorDataset(train_x,train_y)
    model,likelihood = directional_vi.train_gp(dataset,
                        num_inducing=num_inducing,
                        num_directions=num_directions,
                        minibatch_size = minibatch_size,
                        minibatch_dim = num_directions,
                        num_epochs =num_steps, 
                        learning_rate_hypers=learning_rate_hypers,
                        learning_rate_ngd=learning_rate_ngd,
                        inducing_data_initialization=True,
                        use_ngd = use_ngd,
                        use_ciq = use_ciq,
                        lr_sched=lr_sched,
                        mll_type=mll_type,
                        num_contour_quadrature=num_contour_quadrature,
                        verbose=True,
                        )
    return model.double(),likelihood.double()

  def sample_from_gp(model,likelihood,X_cand,n_samples):
    """
    X_cand: 2d torch tensor, points to sample at
    n_samples: int, number of samples to take per point in X_cand
    """
    model.eval()
    likelihood.eval()

    # ensure correct type
    model = model.float()
    likelihood = likelihood.float()
    X_cand = X_cand.float()
    
    n,dim = X_cand.shape
    eval_y = torch.rand(n, dim+1)
    test_dataset = TensorDataset(X_cand, eval_y)
    test_loader = DataLoader(test_dataset, batch_size=gp_eval_batch_size, shuffle=False)
    kwargs = {}
    # means = torch.tensor([0.])
    samples = torch.empty(n_samples, 0)
    with torch.no_grad():
      for x_batch, _ in test_loader:
        if torch.cuda.is_available():
          x_batch = x_batch.cuda()
        derivative_directions = torch.eye(dim)[:model.num_directions]
        derivative_directions = derivative_directions.repeat(x_batch.shape[0],1)
        kwargs['derivative_directions'] = derivative_directions.to(x_batch.device).float()
        preds  = likelihood(model(x_batch,**kwargs))
        samples_cur = preds.sample(torch.Size([n_samples]))[:, ::model.num_directions+1] # shape (n_samples x (dim+1)) 
        # print("samples_cur.shape", samples_cur.shape)
        if torch.cuda.is_available():
          # means = torch.cat([means, preds.mean.detach().cpu()])
          samples = torch.hstack([samples, samples_cur.detach().cpu()])
          # print("samples.shape = ", samples.shape)
        else:
          # means = torch.cat([means, preds.mean])
          samples = torch.hstack([samples, samples_cur])
          # print("samples.shape = ", samples.shape)

    #y_cand = preds.sample(torch.Size([n_samples])) # shape (n_samples x n*(n_dir+1))
    #y_cand = y_cand[:,::model.num_directions+1].t() # shape (n, n_samples)
    
    # means = means[1:]
    # only use mean
    # y_cand = means[::model.num_directions+1].repeat(n_samples,1).t() # (n,n_samples)
  
    ## only use distribution of f(x) to predict (dont use joint covariance with derivatives)
    y_cand = samples.t()

    return y_cand

  from turbo1_grad import *
  # initialize TuRBO
  problem = Turbo1Grad(
        objective,
        lb=turbo_lb,ub=turbo_ub,
        n_init=turbo_n_init,
        max_evals=turbo_max_evals,
        train_gp=train_gp_for_turbo,
        sample_from_gp=sample_from_gp,
        batch_size=turbo_batch_size,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=num_epochs,
        min_cuda=0, # directional_vi.py always runs on cuda if available
        device=turbo_device,
        dtype="float64")
  # optimize
  problem.optimize()
  X_turbo, fX_turbo = problem.X, problem.fX[:,0] # Evaluated points

elif args["model"] == "DSVGP_shared":
  # train
  print("\n\n---TuRBO-Grad with DSVGP_shared in dim {dim}---")
  print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")

  gp_eval_batch_size = 1000
  def train_gp_for_turbo(train_x, train_y, num_steps):
    # expects train_x on unit cube and train_y standardized
    # make a trainable model for TuRBO
    train_x = train_x.float()
    train_y = train_y.float()
    dataset = TensorDataset(train_x,train_y)
    model,likelihood = shared_directional_vi.train_gp(dataset,
                        num_inducing=num_inducing,
                        num_directions=num_directions,
                        minibatch_size = minibatch_size,
                        minibatch_dim = num_directions,
                        num_epochs =num_steps, 
                        learning_rate_hypers=learning_rate_hypers,
                        learning_rate_ngd=learning_rate_ngd,
                        inducing_data_initialization=False,
                        use_ngd = use_ngd,
                        use_ciq = use_ciq,
                        lr_sched=lr_sched,
                        mll_type=mll_type,
                        num_contour_quadrature=num_contour_quadrature,
                        verbose=True,
                        )
    return model.float(),likelihood.float()

  def sample_from_gp(model,likelihood,X_cand,n_samples):
    """
    X_cand: 2d torch tensor, points to sample at
    n_samples: int, number of samples to take per point in X_cand
    """
    model.eval()
    likelihood.eval()

    # ensure correct type
    model = model.float()
    likelihood = likelihood.float()
    X_cand = X_cand.float()
    
    n,dim = X_cand.shape
    eval_y = torch.rand(n, dim+1)
    test_dataset = TensorDataset(X_cand, eval_y)
    test_loader = DataLoader(test_dataset, batch_size=gp_eval_batch_size, shuffle=False)
    kwargs = {}
    # means = torch.tensor([0.])
    samples = torch.empty(n_samples, 0)
    with torch.no_grad():
      for x_batch, _ in test_loader:
        if torch.cuda.is_available():
          x_batch = x_batch.cuda()
        derivative_directions = torch.eye(dim)[:model.num_directions]
        derivative_directions = derivative_directions.repeat(x_batch.shape[0],1)
        kwargs['derivative_directions'] = derivative_directions.to(x_batch.device).float()
        preds  = likelihood(model(x_batch,**kwargs))
        samples_cur = preds.sample(torch.Size([n_samples]))[:, ::model.num_directions+1] # shape (n_samples x (dim+1)) 
        # print("samples_cur.shape", samples_cur.shape)
        if torch.cuda.is_available():
          # means = torch.cat([means, preds.mean.detach().cpu()])
          samples = torch.hstack([samples, samples_cur.detach().cpu()])
          # print("samples.shape = ", samples.shape)
        else:
          # means = torch.cat([means, preds.mean])
          samples = torch.hstack([samples, samples_cur])
          # print("samples.shape = ", samples.shape)

    #y_cand = preds.sample(torch.Size([n_samples])) # shape (n_samples x n*(n_dir+1))
    #y_cand = y_cand[:,::model.num_directions+1].t() # shape (n, n_samples)
    
    # means = means[1:]
    # only use mean
    # y_cand = means[::model.num_directions+1].repeat(n_samples,1).t() # (n,n_samples)
  
    ## only use distribution of f(x) to predict (dont use joint covariance with derivatives)
    y_cand = samples.t()

    return y_cand

  from turbo1_grad import *
  # initialize TuRBO
  problem = Turbo1Grad(
        objective,
        lb=turbo_lb,ub=turbo_ub,
        n_init=turbo_n_init,
        max_evals=turbo_max_evals,
        train_gp=train_gp_for_turbo,
        sample_from_gp=sample_from_gp,
        batch_size=turbo_batch_size,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=num_epochs,
        min_cuda=0, # directional_vi.py always runs on cuda if available
        device=turbo_device,
        dtype="float32")
  # optimize
  problem.optimize()
  X_turbo, fX_turbo = problem.X, problem.fX[:,0] # Evaluated points

elif args["model"] == "SVGP":
  # train
  print("\n\n---TuRBO with Traditional SVGP in dim {dim}---")
  print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")

  from turbo1 import *
  def train_gp_for_turbo(train_x, train_y, use_ard, num_steps, hypers):
    # expects train_x on unit cube and train_y standardized
    # make a trainable model for TuRBO
    train_x = train_x.float()
    train_y = train_y.float()
    dataset = TensorDataset(train_x,train_y)
    model,likelihood = traditional_vi.train_gp(dataset,dim,num_inducing=num_inducing,
                      minibatch_size=minibatch_size,num_epochs=num_steps,use_ngd=use_ngd,
                      use_ciq=use_ciq,learning_rate_hypers=learning_rate_hypers,
                      learning_rate_ngd=learning_rate_ngd,
                      lr_sched=lr_sched,num_contour_quadrature=num_contour_quadrature,
                      mll_type=mll_type,verbose=True)
    return model.double(),likelihood.double()
  
  # initialize TuRBO
  problem = Turbo1(
        objective,
        lb=turbo_lb,ub=turbo_ub,
        n_init=turbo_n_init,
        max_evals=turbo_max_evals,
        train_gp=train_gp_for_turbo,
        batch_size=turbo_batch_size,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=num_epochs,
        min_cuda=0,
        device=turbo_device,
        dtype="float64")
  # optimize
  problem.optimize()
  X_turbo, fX_turbo = problem.X, problem.fX.flatten()  # Evaluated points


elif args["model"] == "TURBO":
    print(f"\n\n---TuRBO with Traditional GP in dim {dim}---")
    from turbo_1 import *
    # initialize TuRBO
    problem = Turbo1(
          objective,
          lb=turbo_lb,ub=turbo_ub,
          n_init=turbo_n_init,
          max_evals=turbo_max_evals,
          batch_size=turbo_batch_size,
          verbose=True,
          use_ard=True,
          max_cholesky_size=2000,
          n_training_steps=num_epochs,
          min_cuda=0,
          device=turbo_device,
          dtype="float64")
    # optimize
    problem.optimize()
    X_turbo, fX_turbo = problem.X, problem.fX.flatten()  # Evaluated points

elif args["model"] == "BO":
  # assert turbo_batch_size == 1
  print(f"\n\n---BO with  GP in dim {dim}---")
  from bo import *
  def train_gp_for_turbo(train_x, train_y, use_ard, num_steps, hypers):
    # expects train_x on unit cube and train_y standardized
    # make a trainable model for TuRBO
    train_x = train_x.float()
    train_y = train_y.float()
    dataset = TensorDataset(train_x,train_y)
    model,likelihood = traditional_vi.train_gp(dataset,dim,num_inducing=num_inducing,
                      minibatch_size=minibatch_size,num_epochs=num_steps,use_ngd=use_ngd,
                      use_ciq=use_ciq,learning_rate_hypers=learning_rate_hypers,
                      learning_rate_ngd=learning_rate_ngd,
                      lr_sched=lr_sched,num_contour_quadrature=num_contour_quadrature,
                      mll_type=mll_type,verbose=False)
    return model.double(),likelihood.double()
    # return model.double()
    
  # initialize TuRBO
  problem = myBO(
        objective,
        lb=turbo_lb,ub=turbo_ub,
        n_init=turbo_n_init,
        train_gp=train_gp_for_turbo,
        max_evals=turbo_max_evals,
        batch_size=turbo_batch_size,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=num_epochs,
        min_cuda=0,
        device=turbo_device,
        dtype="float64")
  # optimize
  problem.optimize()
  X_turbo, fX_turbo = problem.X, problem.fX.flatten()  # Evaluated points

elif args["model"] == "random":
  print(f"\n\n---Random sampling in dim {dim}---")
  from turbo_1 import *
  # use turbo_max_evals as n_init
  assert turbo_batch_size == 1 
  problem = Turbo1(
        objective,
        lb=turbo_lb,ub=turbo_ub,
        n_init=turbo_max_evals,
        max_evals=turbo_max_evals,
        batch_size=turbo_batch_size,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=num_epochs,
        min_cuda=0,
        device=turbo_device,
        dtype="float64")
  # optimize
  problem.optimize()
  X_turbo, fX_turbo = problem.X, problem.fX.flatten()  # Evaluated points
  


def test(data, train=True):
    model.eval()

    correct = 0
    pred = model(data).max(dim=1)[1]

    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / (len(data.y[data.train_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))

# evaluate train and test accuracy on X_turbo
def objective_eval_acc(w):
  model.train()
  model.update_weights(torch.tensor(w, device=turbo_device))
  output = model(data) 
  loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
  train_acc = test(data)
  test_acc = test(data, train=False)
  return train_acc, test_acc

def collect_acc(X):
  train_acc_list, test_acc_list = list(), list()
  for x in X:
    train_acc, test_acc = objective_eval_acc(x)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
  return train_acc_list, test_acc_list

# get the optimum
idx_opt = np.argmin(fX_turbo)
fopt = fX_turbo[idx_opt]
xopt = X_turbo[idx_opt]
print(f"fopt = {fopt}")
train_acc_list, test_acc_list = collect_acc(X_turbo)

if torch.cuda.is_available():
    end.record()
    torch.cuda.synchronize()
    total_time = start.elapsed_time(end)/1e3/60	
    sys.stdout.flush()
else:    
    t2 = time.time_ns()
    total_time = (t2-t1)/1e9/60	

print(f"Total time cost (min):", total_time)

# dump the data
outdata = {}
outdata['X']    = X_turbo
outdata['fX']   = fX_turbo
outdata['train_acc_list'] = train_acc_list
outdata['test_acc_list'] = test_acc_list
outdata['xopt'] = xopt
outdata['fopt'] = fopt
outdata['total_time_min'] = total_time
# add the run params
outdata.update(args)
pickle.dump(outdata,open(data_filename,"wb"))
print(f"Dropped file: {data_filename}")





 
