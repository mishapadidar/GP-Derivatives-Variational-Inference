from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions.synthetic import Branin, SixHumpCamel, StyblinskiTang, Hartmann, SyntheticTestFunction
from torch import Tensor
import torch
from synthetic_functions import Hart_with_deriv

x= Hart_with_deriv()

#t = torch.tensor([[.1, .25, .2, .6, .1, .1], [.2, .1, .3, .1, .2, .4], [.1, .2, .3, .1, .2, .4]])
t = torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]).reshape(1, 6)
res = x.evaluate_true_with_deriv(t)
print(res.shape)
print(res)