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
sys.path.append("../utils")
sys.path.append("../directionalvi")
from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad
from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy
from directional_vi import train_gp, eval_gp
from metrics import MSE
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
print("\n\n---DirectionalGradVGP---")
print(f"Start training with {n} trainig data of dim {dim}")
print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")
t1 = time.time()	
model,likelihood = train_gp(train_dataset,
                      num_inducing=num_inducing,
                      num_directions=num_directions,
                      minibatch_size = minibatch_size,
                      minibatch_dim = num_directions,
                      num_epochs =num_epochs
                      )
t2 = time.time()	

# save the model
# torch.save(model.state_dict(), "../data/test_dvi_basic.model")

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

# # TODO: call some plot util funs here
# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(12,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(test_x[:,0],test_x[:,1],test_y[:,0], color='k')
# ax.scatter(test_x[:,0],test_x[:,1],means[::num_directions+1], color='b')
# plt.title("f(x,y) variational fit; actual curve is black, variational is blue")
# plt.show()
# fig = plt.figure(figsize=(12,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(test_x[:,0],test_x[:,1],test_y[:,1], color='k')
# ax.scatter(test_x[:,0],test_x[:,1],means[1::num_directions+1], color='b')
# plt.title("df/dx variational fit; actual curve is black, variational is blue")
# plt.show()
# fig = plt.figure(figsize=(12,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(test_x[:,0],test_x[:,1],test_y[:,2], color='k')
# ax.scatter(test_x[:,0],test_x[:,1],means[2::num_directions+1], color='b')
# plt.title("df/dy variational fit; actual curve is black, variational is blue")
# plt.show()
