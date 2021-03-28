import numpy as np
import torch

def to_unit_cube(x, lb, ub, g=None):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub, g=None):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1
    xx = x * (ub - lb) + lb
    return xx


def normalize(y, **kwargs):
    '''
    normalize function values and derivatives
    Input: torch tensor storing function values and derivatives
    '''
    if kwargs["derivative"]:
        f = y[..., 0].reshape(len(y),1)
        g = y[..., 1:].reshape(len(y),-1)
        fcopy = np.array(f.flatten())
        sigma = np.std(fcopy)
        f -= np.mean(fcopy)
        f /= sigma
        g /= sigma
        y = torch.cat([f, g], 1) 
    else:
        fcopy = np.array(y.flatten())
        sigma = np.std(fcopy)
        y -= np.mean(fcopy)
        y /= sigma

