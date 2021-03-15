import math
import numpy as np
import torch
import gpytorch
import tqdm
import random
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from test import train_gp

# torch.random.manual_seed(0)
# generate training data
n   = 600
dim = 2
train_x = torch.rand(n,dim)
# f(x) = sin(2pi(x**2+y**2)), df/dx = cos(2pi(x**2+y**2))4pi*x, df/dy = cos(2pi(x**2+y**2))4pi*y
train_y = torch.stack([torch.sin(2*np.pi*(train_x[:,0]**2+train_x[:,1]**2)),
  4*np.pi*train_x[:,0]*torch.cos(2*np.pi*(train_x[:,0]**2+train_x[:,1]**2)),
  4*np.pi*train_x[:,1]*torch.cos(2*np.pi*(train_x[:,0]**2+train_x[:,1]**2))], -1)
# train_y = torch.stack([torch.sin(2*np.pi*train_x[:,0]),2*np.pi*torch.cos(2*np.pi*train_x[:,0]),0.0*train_x[:,1]], -1)


# generate training data
# n   = 100
# dim = 1
# train_x = torch.linspace(0,2*np.pi,n).reshape(n,dim)
# train_y = torch.stack([torch.sin(train_x[:,0]),
#   torch.cos(train_x[:,0])], -1)
# train_y = torch.stack([torch.sin(train_x[:,0])+torch.exp(-3*train_x[:,0]),
#   torch.cos(train_x[:,0]) -3*torch.exp(-3*train_x[:,0])], -1)

num_directions = dim
# train
model,likelihood = train_gp(
                      train_x,
                      train_y,
                      num_inducing=20,
                      num_directions=num_directions,
                      minibatch_size = 100,
                      minibatch_dim = num_directions,
                      num_epochs =10
                      )

# Set into eval mode
model.eval()
likelihood.eval()
for param in model.parameters():
  if param.requires_grad:
    print(param.data)

# predict
kwargs = {}
derivative_directions = torch.eye(dim)[:num_directions]
derivative_directions = derivative_directions.repeat(n,1)
kwargs['derivative_directions'] = derivative_directions
preds   = model(train_x, **kwargs).mean.cpu()

# import matplotlib.pyplot as plt
# pred_f  = preds[::dim+1]
# pred_df = preds[1::dim+1]
# plt.plot(train_x.flatten(),train_y.flatten()[::dim+1],'r-',linewidth=2,label='true f(x)')
# plt.plot(train_x.flatten(),pred_f.detach().numpy(),'b-',linewidth=2,label='variational f(x)')
# plt.plot(train_x.flatten(),train_y.flatten()[1::dim+1],'r--',linewidth=2,label='true df/dx')
# plt.plot(train_x.flatten(),pred_df.detach().numpy(),'b--',linewidth=2,label='variational df/dx')
# plt.legend()
# plt.tight_layout()
# plt.title("Variational GP Predictions with Learned Derivatives")
# plt.show()


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_x[:,0],train_x[:,1],train_y[:,0], color='k')
ax.scatter(train_x[:,0],train_x[:,1],preds.detach().numpy()[::num_directions+1], color='b')
plt.title("f(x,y) variational fit; actual curve is black, variational is blue")
plt.show()
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_x[:,0],train_x[:,1],train_y[:,1], color='k')
ax.scatter(train_x[:,0],train_x[:,1],preds.detach().numpy()[1::num_directions+1], color='b')
plt.title("df/dx variational fit; actual curve is black, variational is blue")
plt.show()
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_x[:,0],train_x[:,1],train_y[:,2], color='k')
ax.scatter(train_x[:,0],train_x[:,1],preds.detach().numpy()[2::dim+1], color='b')
plt.title("df/dy variational fit; actual curve is black, variational is blue")
plt.show()
