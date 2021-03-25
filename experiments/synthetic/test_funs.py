import math
import torch
from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions.synthetic import Branin
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
        val = val.unsqueeze(-1)
        b = 5.1 / (4 * math.pi ** 2)
        c = 5 / math.pi
        t = 1 / (8 * math.pi)
        grad_x2 = 2 * (X[..., 1] - b * X[..., 0] ** 2 + c * X[..., 0] - 6)
        t2 = -2 * b * X[..., 0] + c
        t3 = - 10 * (1 - 1 / (8 * math.pi)) * torch.sin(X[..., 0])
        grad_x1 = grad_x2 * t2 - t3
        grad_x1 = grad_x1.unsqueeze(-1)
        grad_x2 = grad_x2.unsqueeze(-1)
        return torch.cat([val, grad_x1, grad_x2], 1)
