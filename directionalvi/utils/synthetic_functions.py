import math
import torch
import numpy as np
from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions.synthetic import Branin, SixHumpCamel, StyblinskiTang, Hartmann
from torch import Tensor

class Branin_with_deriv(Branin):
    r"""Branin test function.

    Two-dimensional function (usually evaluated on `[-5, 10] x [0, 15]`):

        B(x) = (x_2 - b x_1^2 + c x_1 - r)^2 + 10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`
    B has 3 minimizers for its global minimum at `z_1 = (-pi, 12.275)`,
    `z_2 = (pi, 2.275)`, `z_3 = (9.42478, 2.475)` with `B(z_i) = 0.397887`.
    """
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        val = super().evaluate_true(X)
        val = val.reshape(-1, 1)
        b = 5.1 / (4 * math.pi ** 2)
        c = 5 / math.pi
        t = 1 / (8 * math.pi)
        grad_x2 = 2 * (X[..., 1] - b * X[..., 0] ** 2 + c * X[..., 0] - 6)
        tmp2 = -2 * b * X[..., 0] + c
        tmp3 = - 10 * (1 - t) * torch.sin(X[..., 0])
        grad_x1 = grad_x2 * tmp2 + tmp3
        grad_x1 = grad_x1.reshape(-1, 1)
        grad_x2 = grad_x2.reshape(-1, 1)
        return torch.cat([val, grad_x1, grad_x2], 1)

    def get_bounds(self):
        lb = np.array([item[0] for item in self._bounds])
        ub = np.array([item[1] for item in self._bounds])
        return lb, ub

class SixHumpCamel_with_deriv(SixHumpCamel):
    r"""
    dim = 2
    _bounds = [(-3.0, 3.0), (-2.0, 2.0)]
    SixHumpCamel test function.
      (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2
            + x1 * x2
            + (4 * x2 ** 2 - 4) * x2 ** 2
     """
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        d = X.shape[-1]     
        x1 = X[..., 0]
        x2 = X[..., 1]
        val = super().evaluate_true(X)
        val = val.reshape(-1, 1)
        grad_x1 = x2 + 2 * x1 * (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) + (-4.2 * x1 + 4*x1**3 / 3) * x1 ** 2
        grad_x2 = x1 + 2*x2*(4 * x2 ** 2 - 4) + (8 * x2) * x2 ** 2
        grad_x1 = grad_x1.reshape(-1, 1)
        grad_x2 = grad_x2.reshape(-1, 1)
        return torch.cat([val, grad_x1, grad_x2], 1)

    def get_bounds(self):
            lb = np.array([item[0] for item in self._bounds])
            ub = np.array([item[1] for item in self._bounds])
            return lb, ub


class StyblinskiTang_with_deriv(StyblinskiTang):
    r"""StyblinskiTang test function.

    d-dimensional function (usually evaluated on the hypercube [-5, 5]^d):

    H(x) = 0.5 * sum_{i=1}^d (x_i^4 - 16 * x_i^2 + 5 * x_i)

    H has a single global mininimum H(z) = -39.166166 * d at z = [-2.903534]^d
    """
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        d = X.shape[-1]
        #print("d is: ", d)
        val = super().evaluate_true(X)
        val = val.reshape(-1, 1) #make last dimension 1
        #print("init val is: ", val)
        for i in range(d):
            cur_grad = 0.5*(4* X[..., i] ** 3 - 32 * X[..., i] + 5)
            # cur_grad = cur_grad.unsqueeze(-1)
            cur_grad = cur_grad.reshape(-1, 1)
            val = torch.cat([val, cur_grad], 1)
        return val
    
    def get_bounds(self):
        lb = np.array([item[0] for item in self._bounds])
        ub = np.array([item[1] for item in self._bounds])
        return lb, ub

class Hart_with_deriv(Hartmann):
    r"""Hartmann synthetic test function.

        Most commonly used is the six-dimensional version (typically evaluated on [0, 1]^6):

        H(x) = - sum_{i=1}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij)**2 )
    """
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        ALPHA = self.ALPHA
        A = self.A 
        P = self.P
        
        d = X.shape[-1]
        #print("d is: ", d)
        val = super().evaluate_true(X)
        val = val.unsqueeze(-1) #make last dimension 1

        for j in range(d):
            for i in range(4):
                cur_grad = ALPHA[i] * exp(-1)
        pass

    def get_bounds(self):
        lb = np.array([item[0] for item in self._bounds])
        ub = np.array([item[1] for item in self._bounds])
        return lb, ub