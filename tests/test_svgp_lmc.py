from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import math
import time
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../directionalvi/utils")
sys.path.append("../directionalvi")
from utils.metrics import MSE
import svgp_lmc 
import testfun

n  = 600
n_test = 1000
dim = 2
num_inducing = 20
minibatch_size = int(n/2)
num_epochs = 400
num_tasks = dim+1
num_latents = num_tasks

train_x = torch.rand(n,dim)
test_x = torch.rand(n_test,dim)
train_y = testfun.f(train_x, deriv=True)
test_y = testfun.f(test_x, deriv=True)

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)


print("\n\n---SVGP LMC---")
print(f"Start training with {n} trainig data of dim {dim}")
print(f"VI setups: {num_inducing} inducing points for each task")

args={"verbose":True}

# model training
t1 = time.time_ns()	
model,likelihood = svgp_lmc.train_gp(train_dataset,num_inducing=num_inducing,
                                            minibatch_size=minibatch_size,num_epochs=num_epochs,
                                            learning_rate=0.01,
                                            use_ngd=False,
                                            use_ciq=False,
                                            lr_sched=None,
                                            mll_type="ELBO",
                                            watch_model=False, **args)
t2 = time.time_ns()	
means, variances = svgp_lmc.eval_gp(test_dataset,model,likelihood,minibatch_size=n_test)
t3 = time.time_ns()	

# compute MSE
test_mse = MSE(test_y[:,0].cpu(),means)
# compute mean negative predictive density
test_nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_y[:,0].cpu()).mean()
print(f"At {n_test} testing points, MSE: {test_mse:.4e}, nll: {test_nll:.4e}")
print(f"Training time: {(t2-t1)/1e9:.2f} sec, testing time: {(t3-t2)/1e9:.2f} sec")

