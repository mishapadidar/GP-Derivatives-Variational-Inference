import os
import sys
sys.path.append("../")
from synthetic_functions import *
import torch
import copy
from rescale import *

def comp_err_deriv(fun,x,h):
    '''
    test derivative at x using finite difference
    Inputs:
        fun::callable  function handle, returns function values and derivatives
        x::tensor      one testing position
        h:float        finite difference step size
    '''
    n = x.shape[0]
    d = x.shape[1]
    y = fun(x)
    g_true = y[:,1:]
    id_mat = torch.eye(d)
    error = torch.zeros(n)
    g_fd_set = torch.zeros((n,d))
    for j in range(n):
        g_fd = torch.zeros(d)
        for i in range(d):
            xph = copy.deepcopy(x[j])
            xph[i] = x[j][i] + h
            g_fd[i] = (fun(xph.reshape(1, d))[0][0] - fun(x[j].reshape(1, d))[0][0])/h
        g_fd_set[j,:] = g_fd
        error[j] = (g_fd - g_true[j]).abs().max()
    return error.max(), g_fd_set, g_true

def test_fun_val(test_fun, n):
    try:
        dim = test_fun.dim
    except err:
        dim = 3
    x = torch.rand(n, dim)
    lb, ub = test_fun.get_bounds()
    x = from_unit_cube(x, lb, ub)
    y = test_fun.evaluate_true_with_deriv(x)
    err_fun = (y[:,0] - test_fun.evaluate_true(x)).abs().max()
    return err_fun
    
def test_deriv(test_fun, n):
    # test derivative
    try:
        dim = test_fun.dim
    except err:
        dim = 3
    lb, ub = test_fun.get_bounds()
    x = torch.rand(n, dim)
    x = from_unit_cube(x, lb, ub)
    err, g_fd_set, g_true = comp_err_deriv(test_fun.evaluate_true_with_deriv,x,1e-6)
    return err, g_fd_set, g_true


#branin
test_fun_name = "Branin"
test_fun = eval(f"{test_fun_name}_with_deriv")()
err_fun = test_fun_val(test_fun, 10)
print(f"\nFor {test_fun_name}, error in function values is {err_fun:.4e}.")
err_deriv, _, _ = test_deriv(test_fun, 300)
print(f"For {test_fun_name}, error in derivatives is {err_deriv:.4e}.")



# stytang 
# print("stytang")
# st = StyblinskiTang_with_deriv()
# w = torch.rand(5, 2)
# y = st.evaluate_true_with_deriv(w)
# print(y)
test_fun_name = "StyblinskiTang"
test_fun = eval(f"{test_fun_name}_with_deriv")()
err_fun = test_fun_val(test_fun, 10)
print(f"\nFor {test_fun_name}, error in function values is {err_fun:.4e}.")
err_deriv, _, _ = test_deriv(test_fun, 300)
print(f"For {test_fun_name}, error in derivatives is {err_deriv:.4e}.")



# #six_hump_camel
# print("six hump camel")
# cc = SixHumpCamel_with_deriv()
# w = torch.rand(6, 2)
# y = cc.evaluate_true_with_deriv(w)
# print(y)
test_fun_name = "SixHumpCamel"
test_fun = eval(f"{test_fun_name}_with_deriv")()
err_fun = test_fun_val(test_fun, 10)
print(f"\nFor {test_fun_name}, error in function values is {err_fun:.4e}.")
err_deriv, _, _ = test_deriv(test_fun, 300)
print(f"For {test_fun_name}, error in derivatives is {err_deriv:.4e}.")


# test_fun_name = "Welch"
# test_fun = eval(f"{test_fun_name}_with_deriv")()
# err_fun = test_fun_val(test_fun, 10)
# print(f"\nFor {test_fun_name}, error in function values is {err_fun:.4e}.")
# err_deriv, _, _ = test_deriv(test_fun, 300)
# print(f"For {test_fun_name}, error in derivatives is {err_deriv:.4e}.")


#Hartmann
test_fun_name = "Hartmann"
test_fun = eval(f"{test_fun_name}_with_deriv")()
err_fun = test_fun_val(test_fun, 10)
print(f"\nFor {test_fun_name}, error in function values is {err_fun:.4e}.")
err_deriv, _, _ = test_deriv(test_fun, 300)
print(f"For {test_fun_name}, error in derivatives is {err_deriv:.4e}.")


