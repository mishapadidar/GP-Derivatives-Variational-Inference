import math
import numpy as np
import torch
import gpytorch
import tqdm
import random
import time
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
sys.path.append("../")
sys.path.append("../../directionalvi/utils")
sys.path.append("../../directionalvi")
import directional_vi 
import traditional_vi
import grad_svgp
from csv_dataset import csv_dataset
from metrics import MSE
import pickle
from scipy.io import loadmat
from scipy.sparse import diags


# load a pickle with the run params
args = sys.argv
param_filename = args[1]
run_params = pickle.load(open(param_filename,"rb"))
num_inducing   =run_params['num_inducing']
num_directions =run_params['num_directions'] 
minibatch_size =run_params['minibatch_size'] 
num_epochs     =run_params['num_epochs']
verbose           =run_params['verbose']
inducing_data_initialization =run_params['inducing_data_initialization'] 
use_ngd =run_params['use_ngd']
use_ciq =run_params['use_ciq']
num_contour_quadrature= run_params['num_contour_quadrature']
learning_rate_hypers = run_params['learning_rate_hypers']
learning_rate_ngd    = run_params['learning_rate_ngd']
lr_gamma    = run_params['lr_gamma']
lr_benchmarks = run_params['lr_benchmarks']
lr_sched = run_params['lr_sched']
mll_type = run_params['mll_type']
seed     = run_params['seed']
base_name = run_params['base_name']
mode = run_params['mode']
turbo_lb = run_params['turbo_lb']
turbo_ub =  run_params['turbo_ub']
turbo_n_init =  run_params['turbo_n_init']
turbo_max_evals = run_params['turbo_max_evals']
turbo_batch_size =   run_params['turbo_batch_size']
gamma_mnist = run_params['gamma_mnist']

# make the learning rate schedule
assert lr_sched in [None, "MultiStepLR", "LambdaLR"], "Not a valid choice of lr_sched"
if lr_sched is None:
  pass
elif lr_sched == "MultiStepLR":
  def lr_sched(epoch):
    a = np.sum(lr_benchmarks < epoch)
    # lr_gamma should be > 1
    return (lr_gamma)**a
elif lr_sched == "LambdaLR":
  lr_sched = lambda epoch: 1./(1+lr_gamma*epoch)

# set the seed
torch.random.manual_seed(seed)

# output file names
data_dir = "./output/"
model_filename = data_dir + "model_"+ base_name + ".model"
data_filename  = data_dir + "data_" + base_name + ".pickle"
if os.path.exists(data_dir) is False:
  os.mkdir(data_dir)

if mode == "DSVGP" or mode == "GradSVGP": deriv=True
elif mode == "SVGP": deriv = False

# objective
def svm_loss(z,X,Y,gamma=1.0,deriv=True):
  dim_z = len(z)
  zer = np.zeros_like(Y)
  A = diags(Y)@X
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
    # storage
    fg = np.zeros(dim_z+1)
    # store it 
    fg[0]  = fx
    fg[1:] = np.copy(gx)
  else:   
    # store it 
    fg  = fx
  return fg

# load data
ff = loadmat("mnist.mat")
X_mnist = ff['data'][0,0][1].tocsr()
Y_mnist = ff['data'][0,0][0].astype(float).flatten()
dim_mnist = np.shape(X_mnist)[1]
dim = dim_mnist + 1

# wrap the objective
def myObj(w):
  return svm_loss(w,X_mnist,Y_mnist,gamma_mnist,deriv=deriv)

if torch.cuda.is_available():
  turbo_device = 'cuda'
else:
  turbo_device = 'cpu'

if mode == "DSVGP":
  # train
  print("\n\n---DirectionalGradVGP---")
  print(f"Start training with {n} trainig data of dim {dim}")
  print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")
  t1 = time.time()	
  model,likelihood = directional_vi.train_gp(train_dataset,
                        num_inducing=num_inducing,
                        num_directions=num_directions,
                        minibatch_size = minibatch_size,
                        minibatch_dim = num_directions,
                        num_epochs =num_epochs, 
                        learning_rate_hypers=learning_rate_hypers,
                        learning_rate_ngd=learning_rate_ngd,
                        inducing_data_initialization=inducing_data_initialization,
                        use_ngd = use_ngd,
                        use_ciq = use_ciq,
                        lr_sched=lr_sched,
                        mll_type=mll_type,
                        num_contour_quadrature=num_contour_quadrature,
                        tqdm=tqdm,
                        )
  t2 = time.time()	
  train_time = t2 - t1
  
  # save the model
  torch.save(model.state_dict(),model_filename)
  
  # test
  means, variances = directional_vi.eval_gp(test_dataset,model,likelihood,
                              num_directions=num_directions,
                              minibatch_size=minibatch_size,
                              minibatch_dim=num_directions)
  t3 = time.time()	
  test_time = t3 - t2

  # only keep the function values
  means = means[::num_directions+1]
  variances = variances[::num_directions+1]

elif mode == "SVGP":
  # train
  print("\n\n---TuRBO with Traditional SVGP in dim {dim}---")
  print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")

  from turbo1 import *
  def train_gp_for_turbo(train_x, train_y, use_ard, num_steps, hypers):
    # expects train_x on unit cube and train_y standardized
    # make a trainable model for TuRBO
    train_x = train_x.float()
    train_y = train_y.float()
    dataset = TensorDataset(train_x,train_y)
    model,likelihood = traditional_vi.train_gp(dataset,dim,num_inducing=num_inducing,
                       minibatch_size=minibatch_size,num_epochs=num_steps,use_ngd=use_ngd,
                       use_ciq=use_ciq,learning_rate_hypers=learning_rate_hypers,
                       learning_rate_ngd=learning_rate_ngd,
                       lr_sched=lr_sched,num_contour_quadrature=num_contour_quadrature,
                       mll_type=mll_type,verbose=verbose)
    return model.double(),likelihood.double()
  
  # initialize TuRBO
  problem = Turbo1(
        myObj,
        lb=turbo_lb,ub=turbo_ub,
        n_init=turbo_n_init,
        max_evals=turbo_max_evals,
        train_gp=train_gp_for_turbo,
        batch_size=turbo_batch_size,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=num_epochs,
        min_cuda=1024,
        device=turbo_device,
        dtype="float64")
  # optimize
  problem.optimize()
  X_turbo, fX_turbo = problem.X, problem.fX.flatten()  # Evaluated points
  

elif mode == "GradSVGP":
  # train
  print("\n\n---Grad SVGP---")
  print(f"Start training with {n} training data of dim {dim}")
  print(f"VI setup: {num_inducing} inducing points, {num_directions} inducing directions")
  t1 = time.time()	
  model,likelihood = grad_svgp.train_gp(train_dataset,dim,
                                            num_inducing=num_inducing,
                                            minibatch_size=minibatch_size,
                                            num_epochs=num_epochs,
                                            use_ngd=use_ngd,
                                            use_ciq=use_ciq,
                                            learning_rate_hypers=learning_rate_hypers,
                                            learning_rate_ngd=learning_rate_ngd,
                                            lr_sched=lr_sched,
                                            num_contour_quadrature=num_contour_quadrature,
                                            mll_type=mll_type,
                                            tqdm=False)
  t2 = time.time()	
  train_time = t2 - t1
  
  # save the model
  torch.save(model.state_dict(),model_filename)
  
  # test
  means, variances = grad_svgp.eval_gp(test_dataset,model,likelihood,
                                            num_inducing=num_inducing,
                                            minibatch_size=n_test)
  t3 = time.time()	
  test_time = t3 - t2

  # only keep the function values
  means = means[::dim+1]
  variances = variances[::dim+1]
  

# get the optimum
idx_opt = np.argmin(fX_turbo)
fopt = fX_turbo[idx_opt]
xopt = X_turbo[idx_opt]
print(f"fopt = {fopt}")

# dump the data
outdata = {}
outdata['X']    = X_turbo
outdata['fX']   = fX_turbo
outdata['xopt'] = xopt
outdata['fopt'] = fopt
# add the run params
outdata.update(run_params)
pickle.dump(outdata,open(data_filename,"wb"))
print(f"Dropped file: {data_filename}")
