  #!/usr/bin/env python3
import torch

from gpytorch.lazy.kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad

torch.manual_seed(10)
# generate training data
n1   = 100
n2   = n1
dim = 2
# generate 1d data in 2d
train_x = (torch.linspace(0,2,n1)* (torch.eye(dim)[:1].repeat(n1,1)).T ).T
train_x2 = train_x
# train_x2 = torch.rand(n2,dim)

# set directions
n_directions = 2
# v1 = torch.eye(dim)[:n_directions]
v1 = torch.rand(n_directions,dim)
v1 = v1.repeat(n1,1)
# v1 = torch.rand(n_directions*n1,dim)

# v2 = torch.eye(dim)[:n_directions]
v2 = torch.rand(n_directions,dim)
v2 = v2.repeat(n2,1)
# v2 = torch.rand(n_directions*n2,dim)

# normalize
v1 = (v1.T/torch.norm(v1,dim=1)).T
v2 = (v2.T/torch.norm(v2,dim=1)).T
# setup kernel
k = RBFKernelDirectionalGrad()
params = {'v1':v1,'v2':v2}
K = k(train_x,train_x2, **params)
print(K.detach().numpy().shape)

# finite difference step size
h = 1e-4
# choose the derivative directions
v1_idx = 1  # the index of the v1 vector to take directional derivative
v_deriv = v1[v1_idx::n_directions]
# compute jacobian
Kplus = k(train_x+h*v_deriv,train_x2, **params)
Kmin  = k(train_x-h*v_deriv,train_x2, **params)
jac_v1 = (Kplus.detach().numpy() - Kmin.detach().numpy())/2/h

v2_idx=0 # the index of the v2 vector to take directional derivative
v_deriv = v2[v2_idx::n_directions]
Kplus = k(train_x,train_x2+h*v_deriv, **params)
Kmin  = k(train_x,train_x2-h*v_deriv, **params)
jac_v2 = (Kplus.detach().numpy() - Kmin.detach().numpy())/2/h

import matplotlib.pyplot as plt
plt.plot(K.detach().numpy()[0,::n_directions+1], label='kernel')
plt.plot(K.detach().numpy()[v1_idx+1::n_directions+1,0], label='v1 directional grad')
plt.plot(jac_v1[::n_directions+1,0], label='v1 finite diff')
plt.plot(K.detach().numpy()[0,v2_idx+1::n_directions+1], label='v2 directional grad')
plt.plot(jac_v2[0,::n_directions+1], label='v2 finite diff')


# # verify against RBFKernelGrad
# from gpytorch.kernels import RBFKernelGrad
# kk = RBFKernelGrad()
# KK = kk(train_x,train_x2)
# KKplus = kk(train_x+h*v_deriv,train_x2)
# # KKplus = kk(train_x,train_x2+h*v_deriv)
# jac2_v1 = (KKplus.detach().numpy() - KK.detach().numpy())/h
# plt.plot(KK.detach().numpy()[0,::dim+1], label='kernel gpytorch')
# plt.plot(jac2[0,::dim+1], label='v2 finite diff on gpytorch')
# plt.plot(jac2[::dim+1,0], label='v1 finite diff on gpytorch')

# plt.plot((KK.detach().numpy()[0,1::dim+1]), label='gpytorch e1 directional grad')
plt.legend()
plt.show()
