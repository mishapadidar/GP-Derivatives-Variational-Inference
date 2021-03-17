import torch
import numpy as np

def f(x, deriv=True):
  # f(x) = sin(2pi(x**2+y**2)), df/dx = cos(2pi(x**2+y**2))4pi*x
  fx = torch.sin(2*np.pi*torch.sum(x**2,dim=1))
  gx = 4*np.pi*( torch.cos(2*np.pi*torch.sum(x**2,dim=1)) * x.T).T
  fx = fx.reshape(len(x),1)
  if deriv:
    return torch.cat([fx,gx],1)
  else:   
    return fx.squeeze(axis=1)
