import numpy as np
import torch
import gpytorch
import time
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
sys.path.append("../")
sys.path.append("../../directionalvi/utils")
sys.path.append("../../directionalvi")
import directional_vi 
import traditional_vi
import pickle



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
dim = run_params['dim']

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

if mode == "DSVGP": deriv=True
elif mode == "SVGP" or mode == "Vanilla": deriv = False

# wrap the objective
from rover import *
def myObj(u):
  if deriv==True:
    # stack it
    fg = np.zeros(len(u)+1)
    fg[0] = rover_obj(u)
    fg[1:] = np.copy(rover_grad(u))
    return fg
  else:
    return rover_obj(u)

if torch.cuda.is_available():
  turbo_device = 'cuda'
else:
  turbo_device = 'cpu'

if mode == "DSVGP":
  # train
  print(f"\n\n---TuRBO-Grad with DSVGP in dim {dim}---")
  print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")

  #from turbo1_grad_linesearch import *
  from turbo1_grad import *
  def train_gp_for_turbo(train_x, train_y, use_ard, num_steps, hypers):
    # expects train_x on unit cube and train_y standardized
    # make a trainable model for TuRBO
    train_x = train_x.float()
    train_y = train_y.float()
    dataset = TensorDataset(train_x,train_y)
    model,likelihood = directional_vi.train_gp(dataset,
                        num_inducing=num_inducing,
                        num_directions=num_directions,
                        minibatch_size = minibatch_size,
                        minibatch_dim = num_directions,
                        num_epochs =num_steps, 
                        learning_rate_hypers=learning_rate_hypers,
                        learning_rate_ngd=learning_rate_ngd,
                        inducing_data_initialization=inducing_data_initialization,
                        use_ngd = use_ngd,
                        use_ciq = use_ciq,
                        lr_sched=lr_sched,
                        mll_type=mll_type,
                        num_contour_quadrature=num_contour_quadrature,
                        verbose=verbose,
                        )
    return model.double(),likelihood.double()

  def sample_from_gp(model,likelihood,X_cand,n_samples):
    """
    X_cand: 2d torch tensor, points to sample at
    n_samples: int, number of samples to take per point in X_cand
    """
    model.eval()
    likelihood.eval()

    # ensure correct type
    model = model.float()
    likelihood = likelihood.float()
    X_cand = X_cand.float()
    
    n,dim = X_cand.shape
    kwargs = {}
    derivative_directions = torch.eye(dim)[:model.num_directions]
    derivative_directions = derivative_directions.repeat(n,1)
    kwargs['derivative_directions'] = derivative_directions.to(X_cand.device).float()
    preds  = likelihood(model(X_cand,**kwargs))
    y_cand = preds.sample(torch.Size([n_samples])) # shape (n_samples x n*(n_dir+1))
    y_cand = y_cand[:,::model.num_directions+1].t() # shape (n, n_samples)

    # only use mean
    #y_cand = preds.mean.repeat(n_samples,1).t() # (n,n_samples)

    ## only use distribution of f(x) to predict (dont use joint covariance with derivatives)
    #mean  = preds.mean[::num_directions+1]
    #var  = preds.variance[::num_directions+1] # could have used covariance for f(x) too
    #mvn  = gpytorch.distributions.MultivariateNormal(mean,torch.diag(var))
    #y_cand = mvn.sample(torch.Size([n_samples])).t() # shape (n x n_samples)

    return y_cand

  
  # initialize TuRBO
  problem = Turbo1Grad(
        myObj,
        lb=turbo_lb,ub=turbo_ub,
        n_init=turbo_n_init,
        max_evals=turbo_max_evals,
        train_gp=train_gp_for_turbo,
        sample_from_gp=sample_from_gp,
        batch_size=turbo_batch_size,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=num_epochs,
        min_cuda=0, # directional_vi.py always runs on cuda if available
        device=turbo_device,
        dtype="float64")
  # optimize
  problem.optimize()
  X_turbo, fX_turbo = problem.X, problem.fX[:,0] # Evaluated points

elif mode == "SVGP":
  # train
  print(f"\n\n---TuRBO with Traditional SVGP in dim {dim}---")
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
        min_cuda=0,
        device=turbo_device,
        dtype="float64")
  # optimize
  problem.optimize()
  X_turbo, fX_turbo = problem.X, problem.fX.flatten()  # Evaluated points
  
elif mode == "Vanilla":
  # train
  print(f"\n\n---Vanilla TuRBO in dim {dim}---")

  from turbo1_vanilla import *

  # initialize TuRBO
  problem = Turbo1(
        myObj,
        lb=turbo_lb,ub=turbo_ub,
        n_init=turbo_n_init,
        max_evals=turbo_max_evals,
        batch_size=turbo_batch_size,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=num_epochs,
        min_cuda=0,
        device=turbo_device,
        dtype="float64")

  # optimize
  problem.optimize()
  X_turbo, fX_turbo = problem.X, problem.fX.flatten()  # Evaluated points

  

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
