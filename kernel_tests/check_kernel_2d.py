#!/usr/bin/env python3
import torch
import matplotlib.pyplot as plt
from gpytorch.lazy.kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad
import numpy as np

torch.manual_seed(100)
# generate training data
n1   = 100
n2   = n1
dim = 2


## ===========================================
# Test 1: Varying Positions, 1 direction
# A single direction is used at all points
## ===========================================

# generate 1d data in 2d
train_x = torch.zeros(n1,dim)
train_x2 = train_x
a = 0
b = 10.0
xx,yy = torch.meshgrid(torch.linspace(a,b,int(np.sqrt(n2))),torch.linspace(a,b,int(np.sqrt(n2))))
train_x2[:,0] = xx.flatten()
train_x2[:,1] = yy.flatten()
#train_x2 = torch.rand(n2,dim)

# set directions
n_directions = 1
# v1 = torch.tensor([[ 0.5632, -0.1039]])
# v1 = torch.tensor([[1.0,0.0]])
# v1 = v1.repeat(n1,1)
v1 = torch.randn(n1,2)
#print(v1)
v2 = v1
# normalize
v1 = (v1.T/torch.norm(v1,dim=1)).T
v2 = (v2.T/torch.norm(v2,dim=1)).T

# setup kernel
k = RBFKernelDirectionalGrad()
params = {'v1':v1,'v2':v2}
K = k(train_x,train_x2, **params).detach().numpy()

# compute jacobian on right derivative
h = 1e-4
Kplus = k(train_x,train_x2+h*v2, **params)
Kmin  = k(train_x,train_x2-h*v2, **params)
jac_v2 = ((Kplus.detach().numpy() - Kmin.detach().numpy())/2/h)[0,::n_directions+1]

# scale the directions to the get derivative vector
true_dd_v2 = np.diag(jac_v2) @ v2.detach().numpy()
kern_dd_v2 = np.diag(K[0,1::n_directions+1]) @ v2.detach().numpy()
# compute the difference
diff_dd_v2 = kern_dd_v2 - true_dd_v2
print(diff_dd_v2)

# plot it 
#plt.quiver(xx.flatten(),yy.flatten(),true_dd_v2[:,0],true_dd_v2[:,1],color='k')
#plt.quiver(xx.flatten(),yy.flatten(),kern_dd_v2[:,0],kern_dd_v2[:,1],color='b')
plt.quiver(train_x2[:,0].flatten(),train_x2[:,1].flatten(),diff_dd_v2[:,0],diff_dd_v2[:,1],np.linalg.norm(diff_dd_v2))
plt.colorbar()
plt.show()

## ===========================================
# Test 1: Varying Positions, 1 direction
# A single direction is used at all points
## ===========================================

# generate 1d data in 2d
train_x = torch.zeros(n1,dim)
train_x2 = train_x
a = 0
b = 1.0
xx,yy = torch.meshgrid(torch.linspace(a,b,int(np.sqrt(n2))),torch.linspace(a,b,int(np.sqrt(n2))))
train_x2[:,0] = xx.flatten()
train_x2[:,1] = yy.flatten()
#train_x2 = torch.rand(n2,dim)

# set directions
n_directions = 1
# v1 = torch.tensor([[ 0.5632, -0.1039]])
# v1 = torch.tensor([[1.0,0.0]])
# v1 = v1.repeat(n1,1)
v1 = torch.randn(n1,2)
#print(v1)
v2 = v1
# normalize
v1 = (v1.T/torch.norm(v1,dim=1)).T
v2 = (v2.T/torch.norm(v2,dim=1)).T

# setup kernel
k = RBFKernelDirectionalGrad()
params = {'v1':v1,'v2':v2}
K = k(train_x,train_x2, **params).detach().numpy()

# compute jacobian on right derivative
h = 1e-3
Kplus = k(train_x+h*v1,train_x2, **params)
Kmin  = k(train_x-h*v1,train_x2, **params)
jac_v1 = ((Kplus.detach().numpy() - Kmin.detach().numpy())/2/h)[0,::n_directions+1]

# scale the directions to the get derivative vector
true_dd_v1 = np.diag(jac_v1) @ v1.detach().numpy()
kern_dd_v1 = np.diag(K[1::n_directions+1,0]) @ v1.detach().numpy()
# compute the difference
diff_dd_v1 = kern_dd_v1 - true_dd_v1
print(diff_dd_v1)

# plot it 
#plt.quiver(xx.flatten(),yy.flatten(),true_dd_v2[:,0],true_dd_v2[:,1],color='k')
#plt.quiver(xx.flatten(),yy.flatten(),kern_dd_v2[:,0],kern_dd_v2[:,1],color='b')
plt.quiver(train_x2[:,0].flatten(),train_x2[:,1].flatten(),diff_dd_v1[:,0],diff_dd_v1[:,1],np.linalg.norm(diff_dd_v1))
plt.colorbar()
plt.show()