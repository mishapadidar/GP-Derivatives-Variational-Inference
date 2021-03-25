import math
import numpy as np
import torch
import gpytorch
import tqdm
import random
import time
import sys
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
sys.path.append("../")
sys.path.append("../directionalvi/utils")
sys.path.append("../directionalvi")
from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad
from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy
from directional_vi import eval_gp, train_gp_ngd
from utils.metrics import MSE
import testfun

# data parameters
n   = 600
dim = 2
n_test = 1000

# training params
num_inducing = 20
num_directions = 1
minibatch_size = 200
num_epochs = 1000

# seed
torch.random.manual_seed(0)

# generate training data
train_x = torch.rand(n,dim)
train_y = testfun.f(train_x)
train_dataset = TensorDataset(train_x,train_y)

# testing data
test_x = torch.rand(n_test,dim)
test_y = testfun.f(test_x)
test_dataset = TensorDataset(test_x,test_y)

# train
print("\n\n---DirectionalGradVGP with NGD---")
print(f"Start training with {n} trainig data of dim {dim}")
print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")
t1 = time.time_ns()	
model,likelihood = train_gp_ngd(train_dataset,
                      num_inducing=num_inducing,
                      num_directions=num_directions,
                      minibatch_size = minibatch_size,
                      minibatch_dim = num_directions,
                      num_epochs =num_epochs, tqdm=True
                      )

# save the model
# torch.save(model.state_dict(), "../data/test_dvi_basic.model")
t2 = time.time_ns()	

# test
means, variances = eval_gp( test_dataset,model,likelihood,
                            num_inducing=num_inducing,
                            num_directions=num_directions,
                            minibatch_size=n_test,
                            minibatch_dim=num_directions)
t3 = time.time_ns()	

# compute MSE
test_mse = MSE(test_y[:,0],means[::num_directions+1])
# compute mean negative predictive density
test_nll = -torch.distributions.Normal(means[::num_directions+1], variances.sqrt()[::num_directions+1]).log_prob(test_y[:,0]).mean()
print(f"At {n_test} testing points, MSE: {test_mse:.4e}, nll: {test_nll:.4e}")
print(f"Training time: {(t2-t1)/1e9:.2f} sec, testing time: {(t3-t2)/1e9:.2f} sec")

# TODO: call some plot util funs here