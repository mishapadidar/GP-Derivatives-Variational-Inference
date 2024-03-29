import math
import numpy as np
from scipy.io import loadmat
import torch
import gpytorch
import random
import time
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
sys.path.append("../")
sys.path.append("../../directionalvi/utils")
sys.path.append("../../directionalvi")
import dfree_directional_vi 
import traditional_vi
from csv_dataset import csv_dataset
from metrics import MSE
import pickle


# load a pickle with the run params
args = sys.argv
param_filename = args[1]
run_params = pickle.load(open(param_filename,"rb"))
num_inducing   =run_params['num_inducing']
num_directions =run_params['num_directions'] 
minibatch_size =run_params['minibatch_size'] 
num_epochs     =run_params['num_epochs']
verbose        =run_params['verbose']
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
data_file = run_params['data_file']
mode = run_params['mode']

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
#data_dir = "./output/"
data_dir = run_params['data_dir']
model_filename = data_dir + "model_"+ base_name + ".model"
data_filename  = data_dir + "data_" + base_name + ".pickle"
if os.path.exists(data_dir) is False:
  os.mkdir(data_dir)

# load data
ff = loadmat(data_file)
X_data = torch.tensor(ff['data'][:,:-1]) # @Leo double check this is right
y_data = torch.tensor(ff['data'][:,-1]) # @Leo double check this is right
n, dim = X_data.shape

# make sure right type
X_data =X_data.float()
y_data =y_data.float()

# standardize data
lb = torch.min(X_data,axis=0)[0]
ub = torch.max(X_data,axis=0)[0]
X_data = (X_data - lb)/(ub-lb)
med = torch.median(y_data)
std = torch.std(y_data)
y_data = (y_data - med)/std

# make a torch dataset
dataset = TensorDataset(X_data,y_data)

# train-test split
n_train = int(0.8*n)
n_test  = n - n_train
train_dataset,test_dataset = torch.utils.data.random_split(dataset,[n_train,n_test])

# make dataloaders
train_loader  = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=n_test, shuffle=False)


if mode == "DSVGP":
  # train
  print("\n\n---DirectionalGradVGP---")
  print(f"Start training with {n} trainig data of dim {dim}")
  print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")
  t1 = time.time()	
  model,likelihood = dfree_directional_vi.train_gp(train_dataset,
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
                        verbose=verbose,
                        )
  t2 = time.time()	
  train_time = t2 - t1
  
  # save the model
  torch.save(model.state_dict(),model_filename)
  
  # test
  means, variances = dfree_directional_vi.eval_gp(test_dataset,model,likelihood,
                              num_directions=num_directions,
                              minibatch_size=minibatch_size,
                              minibatch_dim=num_directions)
  t3 = time.time()	
  test_time = t3 - t2

elif mode == "SVGP":
  # train
  print("\n\n---Traditional SVGP---")
  print(f"Start training with {n} training data of dim {dim}")
  print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")
  t1 = time.time()	
  model,likelihood = traditional_vi.train_gp(train_dataset,dim,
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
                                            verbose=verbose)
  t2 = time.time()	
  train_time = t2 - t1
  
  # save the model
  torch.save(model.state_dict(),model_filename)
  
  # test
  means, variances = traditional_vi.eval_gp(test_dataset,model,likelihood, 
                                            num_inducing=num_inducing,
                                            minibatch_size=n_test)
  t3 = time.time()	
  test_time = t3 - t2


# collect the test function values
test_f = torch.zeros(n_test)
for ii in range(n_test):
    test_f[ii] = test_dataset[ii][1] # function value

# compute MSE
test_mse = MSE(test_f,means)
# compute mean negative predictive density
test_nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_f).mean()
print(f"At {n_test} testing points, MSE: {test_mse:.4e}, nll: {test_nll:.4e}.")
print(f"Training time: {train_time:.2f} sec, testing time: {test_time:.2f} sec")

# dump the data
outdata = {}
outdata['test_mse']   = test_mse
outdata['test_nll']   = test_nll
outdata['train_time'] = train_time
outdata['test_time']  = test_time
# add the run params
outdata.update(run_params)
pickle.dump(outdata,open(data_filename,"wb"))
print(f"Dropped file: {data_filename}")
