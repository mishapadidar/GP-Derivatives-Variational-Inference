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
from grad_svgp import train_gp,eval_gp
import testfun


# setups
n  = 600
n_test = 1000
dim = 2
num_inducing = 20
minibatch_size = int(n/2)
num_epochs = 400
use_ciq = False 
use_ngd = False 
learning_rate_hypers=0.01
learning_rate_ngd=0.1
# lr-schedule
gamma  = 10.0
levels = np.array([20,150,300])
def lr_sched(epoch):
  a = np.sum(levels > epoch)
  return (1./gamma)**a
lr_sched=None
mll_type="PLL"

# seed
torch.random.manual_seed(0)

# trainig and testing data
train_x = torch.rand(n,dim)
test_x = torch.rand(n_test,dim)
train_y = testfun.f(train_x, deriv=True)
test_y = testfun.f(test_x, deriv=True)
if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=n_test, shuffle=False)

print("\n\n---Standard SVGP---")
print(f"Start training with {n} trainig data of dim {dim}")
print(f"VI setups: {num_inducing} inducing points")

args={"verbose":True}

# model training
t1 = time.time_ns()	
model,likelihood = train_gp(train_dataset,dim,
                                            num_inducing=num_inducing,
                                            minibatch_size=minibatch_size,
                                            num_epochs=num_epochs,
                                            use_ngd=use_ngd,
                                            use_ciq=use_ciq,
                                            learning_rate_hypers=learning_rate_hypers,
                                            learning_rate_ngd=learning_rate_ngd,
                                            lr_sched=lr_sched,
                                            tqdm=False, 
                                            mll_type=mll_type,
                                            **args)
t2 = time.time_ns()	
means, variances = eval_gp(test_dataset,model,likelihood, 
                                            num_inducing=num_inducing,
                                            minibatch_size=n_test)
t3 = time.time_ns()	

# compute MSE
#test_mse = MSE(test_y.cpu(),means)
test_mse = MSE(test_y[:,0],means[::dim+1])
# compute mean negative predictive density
test_nll = -torch.distributions.Normal(means[::dim+1], variances.sqrt()[::dim+1]).log_prob(test_y[:,0]).mean()
#test_nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_y.cpu()).mean()
print(f"At {n_test} testing points, MSE: {test_mse:.4e}, nll: {test_nll:.4e}")
print(f"Training time: {(t2-t1)/1e9:.2f} sec, testing time: {(t3-t2)/1e9:.2f} sec")


# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(12,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(train_x[:,0],train_x[:,1],train_y, color='k')
# ax.scatter(train_x[:,0],train_x[:,1],means, color='b')
# plt.title("f(x,y) variational fit; actual curve is black, variational is blue")
# plt.show()
