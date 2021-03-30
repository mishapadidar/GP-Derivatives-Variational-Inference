import os
import sys
sys.path.append("../")
from synthetic_functions import *
import torch

#branin
x = Branin_with_deriv()
w = torch.rand(10, 2)
y = x.evaluate_true_with_deriv(w)
print(y)
z = Branin().evaluate_true(w)
print(z)

# stytang 
print("stytang")
st = StyblinskiTang_with_deriv()
w = torch.rand(5, 2)
y = st.evaluate_true_with_deriv(w)
print(y)