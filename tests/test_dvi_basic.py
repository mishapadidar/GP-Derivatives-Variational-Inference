import math
import numpy as np
import torch
import gpytorch
import tqdm
import random
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.append("../directionalvi")
sys.path.append("../directionalvi/utils")
from directional_vi import train_gp
#from directionalvi.RBFKernelDirectionalGrad import RBFKernelDirectionalGrad
#from directionalvi.DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy
#from directionalvi.directional_vi import train_gp
#from directionalvi.utils.metrics import MSE

def f(x):
  # f(x) = sin(2pi(x**2+y**2)), df/dx = cos(2pi(x**2+y**2))4pi*x
  fx = torch.sin(2*np.pi*torch.sum(x**2,dim=1))
  gx = 4*np.pi*( torch.cos(2*np.pi*torch.sum(x**2,dim=1)) * x.T).T
  fx = fx.reshape(len(x),1)
  return torch.cat([fx,gx],1)

# data parameters
n   = 600
dim = 2
n_test = 1000

# training params
num_inducing = 20
num_directions = dim
minibatch_size = 200
num_epochs = 1000

# seed
# torch.random.manual_seed(0)

# generate training data
train_x = torch.rand(n,dim)
train_y = f(train_x)
train_dataset = TensorDataset(train_x,train_y)

# testing data
test_x = torch.rand(n_test,dim)
test_y = f(test_x)
test_dataset = TensorDataset(test_x,test_y)
test_loader = DataLoader(test_dataset, batch_size=n_test, shuffle=False)


# train
model,likelihood = train_gp(train_dataset,
                      num_inducing=num_inducing,
                      num_directions=num_directions,
                      minibatch_size = minibatch_size,
                      minibatch_dim = num_directions,
                      num_epochs =num_epochs
                      )

# Set into eval mode
model.eval()
likelihood.eval()

# print out the trained parameters
#for param in model.parameters():
#  if param.requires_grad:
#    print(param.data)

# save the model
torch.save(model.state_dict(), "../data/test_dvi_basic.model")


# predict in batches
kwargs = {}
derivative_directions = torch.eye(dim)[:num_directions]
derivative_directions = derivative_directions.repeat(n_test,1)
kwargs['derivative_directions'] = derivative_directions
#means   = model(test_x, **kwargs).mean.cpu()
means = torch.tensor([0.])
test_y = torch.zeros(1,dim+1)
test_x = torch.zeros(1,dim)
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        print(y_batch)
        preds = model(x_batch,**kwargs)
        means = torch.cat([means, preds.mean.cpu()])
        test_y = torch.cat([test_y, y_batch])
        test_x = torch.cat([test_x, x_batch])

means = means[1:]
test_y = test_y[1:]
test_x = test_x[1:]

# compute MSE
test_mse = MSE(test_y[:,0],means[::num_directions+1])
print(f"\nTesting MSE: {test_mse}")

# plot it
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_x[:,0],test_x[:,1],test_y[:,0], color='k')
ax.scatter(test_x[:,0],test_x[:,1],means[::num_directions+1], color='b')
plt.title("f(x,y) variational fit; actual curve is black, variational is blue")
plt.show()
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_x[:,0],test_x[:,1],test_y[:,1], color='k')
ax.scatter(test_x[:,0],test_x[:,1],means[1::num_directions+1], color='b')
plt.title("df/dx variational fit; actual curve is black, variational is blue")
plt.show()
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_x[:,0],test_x[:,1],test_y[:,2], color='k')
ax.scatter(test_x[:,0],test_x[:,1],means[2::num_directions+1], color='b')
plt.title("df/dy variational fit; actual curve is black, variational is blue")
plt.show()
