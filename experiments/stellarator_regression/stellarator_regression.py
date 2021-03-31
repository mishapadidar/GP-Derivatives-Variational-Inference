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
from directional_vi import train_gp, eval_gp
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
tqdm           =run_params['tqdm']
inducing_data_initialization =run_params['inducing_data_initialization'] 
use_ngd =run_params['use_ngd']
use_ciq =run_params['use_ciq']
num_contour_quadrature= run_params['num_contour_quadrature']
learning_rate_hypers = run_params['learning_rate_hypers']
learning_rate_ngd    = run_params['learning_rate_ngd']
lr_gamma    = run_params['lr_gamma']
lr_benchmarks = run_params['lr_benchmarks']
#lr_sched = run_params['lr_sched']
seed     = run_params['seed']
base_name = run_params['base_name']
data_file = run_params['data_file']
mode = run_params['mode']

# make the learning rate schedule
#lr_sched = lambda epoch: 1./(1+lr_gamma*epoch)
def lr_sched(epoch):
  a = np.sum(lr_benchmarks > epoch)
  return (1./lr_gamma)**a

# set the seed
torch.random.manual_seed(seed)

# output file names
data_dir = "./output/"
model_filename = data_dir + base_name + ".model"
data_filename  = data_dir + "data_" + base_name + ".pickle"
if os.path.exists(data_dir) is False:
  os.mkdir(data_dir)

# load a dataset
if mode == "SVGP": gradients = False
else: gradients = True
dataset = csv_dataset(data_file,rescale=True,gradients=gradients)
dim = len(dataset[0][0])
n   = len(dataset)

# train-test split
n_train = int(0.8*n)
n_test  = int(0.2*n)

train_dataset,test_dataset = torch.utils.data.random_split(dataset,[n_train,n_test])


if mode == "DSVGP":
  # train
  print("\n\n---DirectionalGradVGP---")
  print(f"Start training with {n} trainig data of dim {dim}")
  print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")
  t1 = time.time()	
  model,likelihood = train_gp(train_dataset,
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
                        num_contour_quadrature=num_contour_quadrature,
                        tqdm=tqdm,
                        )
  t2 = time.time()	
  train_time = t2 - t1
  
  # save the model
  torch.save(model.state_dict(),model_filename)
  
  # test
  means, variances = eval_gp(test_dataset,model,likelihood,
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
                                            tqdm=False)
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
  if mode == "DSVGP":
    test_f[ii] = test_dataset[ii][1][0] # function value
  elif mode == "SVGP":
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
