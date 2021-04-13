import torch
import pickle
import numpy as np

# objective
def testf(x, deriv=True):
  # f(x) = sin(2pi(x**2+y**2)), df/dx = cos(2pi(x**2+y**2))4pi*x
  fx = torch.sin(2*np.pi*torch.sum(x**2,dim=1))
  gx = 4*np.pi*( torch.cos(2*np.pi*torch.sum(x**2,dim=1)) * x.T).T
  fx = fx.reshape(len(x),1)
  if deriv:
    return torch.cat([fx,gx],1)
  else:   
    return fx.squeeze(axis=1)

n  = 20000
dim = 12
X = torch.rand(n,dim)
Y = testf(X)
d ={}
d['X'] = X
d['Y'] = Y
name = f"./synthetic1_dataset_{n}_points_{dim}_dim.pickle"
pickle.dump(d,open(name,"wb"))
