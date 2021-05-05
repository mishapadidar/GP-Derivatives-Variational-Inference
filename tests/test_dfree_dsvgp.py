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
sys.path.append("../directionalvi/utils")
sys.path.append("../directionalvi")
from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad
#from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy
from dfree_directional_vi import train_gp, eval_gp
from metrics import MSE
import testfun

# data parameters
n   = 600
dim = 2
n_test = 1000

# training params
num_inducing = 20
num_directions = 2
minibatch_size = 200
num_epochs = 400

# seed
torch.random.manual_seed(0)
# use tqdm or just have print statements
tqdm = False
# use data to initialize inducing stuff
inducing_data_initialization = False
# use natural gradients and/or CIQ
use_ngd = False
use_ciq = False
num_contour_quadrature=15
# learning rate
learning_rate_hypers = 0.01
learning_rate_ngd    = 0.1
gamma  = 10.0
#levels = np.array([20,150,300])
#def lr_sched(epoch):
#  a = np.sum(levels > epoch)
#  return (1./gamma)**a
lr_sched = None

# training and testing data
train_x = torch.rand(n,dim)
test_x = torch.rand(n_test,dim)
train_y = testfun.f(train_x, deriv=True)
test_y = testfun.f(test_x, deriv=False)
if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=n_test, shuffle=False)

# train
print("\n\n---DirectionalGradVGP---")
print(f"Start training with {n} trainig data of dim {dim}")
print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")
args={"verbose":True}
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
                      num_contour_quadrature=num_contour_quadrature,
                      tqdm=tqdm,**args
                      )
t2 = time.time()	

# save the model
# torch.save(model.state_dict(), "../data/test_dvi_basic.model")

# test
means, variances = eval_gp( test_dataset,model,likelihood,
                            num_directions=num_directions,
                            minibatch_size=n_test,
                            minibatch_dim=num_directions)
t3 = time.time()	

# compute MSE
test_y = test_y.cpu()
test_mse = MSE(test_y,means)
# compute mean negative predictive density
test_nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_y).mean()
print(f"At {n_test} testing points, MSE: {test_mse:.4e}, nll: {test_nll:.4e}.")
print(f"Training time: {(t2-t1):.2f} sec, testing time: {(t3-t2):.2f} sec")

plot=1
if plot == 1:
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(test_x[:,0],test_x[:,1],test_y, color='k')
    ax.scatter(test_x[:,0],test_x[:,1],means, color='b')
    plt.title("f(x,y) variational fit; actual curve is black, variational is blue")
    plt.show()

