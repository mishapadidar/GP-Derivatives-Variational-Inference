import torch
import pickle
import numpy as np
from scipy.io import loadmat
from scipy.sparse import diags
import sys

# objective
def svm_loss(Z,X,Y,gamma=1.0,deriv=True):
  n,dim_z = np.shape(Z)
  zer = np.zeros_like(Y)
  A = diags(Y)@X

  # storage
  if deriv:
    fg = np.zeros((n,dim_z+1))
  else:
    fg = np.zeros(n)

  for ii in range(n):
    print(f"Evaluation {ii}/{n}")
    sys.stdout.flush()
    z = Z[ii]
    w = z[:-1]
    b = z[-1]
    c = Y*b
    # SVM primal loss w/ squared hinge loss
    fx = np.mean( np.maximum(zer,1-(A @ w - c))**2 ) + gamma*np.sum(w**2)
    if deriv:
      # df/dw
      gx = np.zeros_like(z)
      gx[:-1] = np.mean(diags(2*np.maximum(zer,1-(A @ w- c)))*(-A), axis=0) + gamma*2*w
      # df/db
      gx[-1] = np.mean(2*np.maximum(zer,1-(A @ w- c))*Y, axis=0)
      # store it 
      fg[ii,0]  = fx
      fg[ii,1:] = np.copy(gx)
    else:   
      # store it 
      fg[ii]  = fx
  return fg


# load data
ff = loadmat("mnist.mat")
data = ff['data']
X = data[0,0][1].tocsr()
Y = data[0,0][0].astype(float).flatten()
x_dim = X.shape[1]

# generate random weights
n  = 1000
w_dim = x_dim + 1
W = np.random.rand(n,w_dim)

# compute the svm loss and gradients
gamma = 0.1
F = svm_loss(W,X,Y,gamma,deriv=True)

# save the data
d ={}
d['X'] = torch.tensor(W)
d['Y'] = torch.tensor(F)
name = f"./svm_dataset_{n}_points.pickle"
pickle.dump(d,open(name,"wb"))
