import math
import numpy as np
import torch
import gpytorch
import tqdm
import random
import time
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.append("../")
sys.path.append("../../directionalvi/utils")
sys.path.append("../../directionalvi")
from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad
from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy
from directional_vi import train_gp, eval_gp
from csv_dataset import csv_dataset
from metrics import MSE

# load a dataset
dataset = csv_dataset("../../data/focus_w7x_dataset.csv",rescale=True)
dim = len(dataset[0][0])
n   = len(dataset)

# train-test split
n_train = int(0.8*n)
n_test  = int(0.2*n)

# training params
num_inducing   = 512
num_directions = 2
minibatch_size = 1024
num_epochs     = 400

# seed
torch.random.manual_seed(0)
tqdm = False
# use data to initialize inducing stuff
inducing_data_initialization = False
# use natural gradients and/or CIQ
use_ngd = True 
use_ciq = False 
# learning rate
learning_rate_hypers = 1e-3
learning_rate_ngd    = 1e-3
gamma = 10
lr_sched = lambda epoch: 1.0/(1+gamma*epoch)

train_dataset,test_dataset = torch.utils.data.random_split(dataset,[n_train,n_test])

# train
print("\n\n---DirectionalGradVGP---")
print(f"Start training with {n} trainig data of dim {dim}")
print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")
t1 = time.time()	
model,likelihood = train_gp(train_dataset,
                      num_inducing=num_inducing,
                      num_directions=num_directions,
                      minibatch_size = minibatch_size,
                      minibatch_dim = num_directions,
                      num_epochs =num_epochs, 
                      learning_rate_hypers=learning_rate_hypers,
                      learning_rate_ngd=learning_rate_ngd,
                      inducing_data_initialization=inducing_data_initialization,
                      use_ngd = use_ngd,
                      use_ciq = use_ciq,
                      lr_sched=lr_sched,
                      tqdm=tqdm,
                      )
t2 = time.time()	

# save the model
if use_ciq:
  post = "ciq"
elif use_ngd:
  post = "ngd"
else:
  post = "basic"
model_filename = f"./stell_regress_{num_inducing}_{num_directions}_{post}.model"
torch.save(model.state_dict(),model_filename)

# test
means, variances = eval_gp(test_dataset,model,likelihood,
                            num_directions=num_directions,
                            minibatch_size=minibatch_size,
                            minibatch_dim=num_directions)
t3 = time.time()	

# collect the test function values
test_f = torch.zeros(n_test)
for ii in range(n_test):
  test_f[ii] = test_dataset[ii][1][0] # function value

# compute MSE
test_mse = MSE(test_f,means[::num_directions+1])
# compute mean negative predictive density
test_nll = -torch.distributions.Normal(means[::num_directions+1], variances.sqrt()[::num_directions+1]).log_prob(test_f).mean()
print(f"At {n_test} testing points, MSE: {test_mse:.4e}, nll: {test_nll:.4e}.")
print(f"Training time: {(t2-t1):.2f} sec, testing time: {(t3-t2):.2f} sec")
