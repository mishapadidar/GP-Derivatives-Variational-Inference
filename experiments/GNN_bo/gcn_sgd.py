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


exp_name = args["exp_name"]
num_epochs = args["turbo_max_evals"]
args["model"] = "SGD"
expname_full =  f"{args['dataset']}_{args['model']}_epochs{num_epochs}_exp{exp_name}"
print(expname_full)


# output result file names
data_dir = "./results/"
data_filename  = data_dir + expname_full + ".pickle"
if os.path.exists(data_dir) is False:
  os.mkdir(data_dir)


torch.set_default_dtype(torch.float64)
torch.random.manual_seed(args["seed"])  

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


def train(data, plot=False):
  train_acc_list, test_acc_list, loss_history = list(), list(), list()
  for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    train_acc = test(data)
    test_acc = test(data, train=False)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    loss_history.append(loss.item())
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
            format(epoch, loss.item(), train_acc, test_acc))
  return loss_history, train_acc_list, test_acc_list



# load data for GCN
dataset = "PubMed"
assert args["dataset"] == "PubMed"
dataset = Planetoid(root='/tmp/PubMed', name='PubMed')  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
turbo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(dataset).to(device)
data = dataset[0].to(device)
print("\nDimension of GCN:", model.n_params)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_history, train_acc_list, test_acc_list = train(data)

print("\nFinal parameters of GCN:", )
for name, param in model.named_parameters():
  print(name)
  print(param)
  print(param.min())
  print(param.max())
  print()

# dump the data
outdata = {}
outdata['X']    = None
outdata['fX']   = loss_history
outdata['train_acc_list'] = train_acc_list
outdata['test_acc_list'] = test_acc_list
outdata['xopt'] = None
outdata['fopt'] = min(loss_history)
# add the run params
outdata.update(args)
pickle.dump(outdata,open(data_filename,"wb"))
print(f"Dropped file: {data_filename}")





 
