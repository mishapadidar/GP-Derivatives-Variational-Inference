import torch
import gpytorch
import math
import numpy as np
import sys

class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def train_gp(train_x,train_y,num_epochs=1,lr_hypers=0.01,verbose=True):

  dim = train_x.shape[-1]
  n_tasks = dim + 1
  likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)  # Value + x-derivative + y-derivative
  model = GPModelWithDerivatives(train_x, train_y, likelihood)
  
  # Find optimal model hyperparameters
  model.train()
  likelihood.train()
  
  # Use the adam optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=lr_hypers)  # Includes GaussianLikelihood parameters
  
  # "Loss" for GPs - the marginal log likelihood
  mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
  
  for i in range(num_epochs):
      optimizer.zero_grad()
      output = likelihood(model(train_x))
      loss = -mll(output, train_y)
      loss.backward()
      print(f"Iter {i}, Loss: {loss.item()}")
      sys.stdout.flush()
      optimizer.step()

  print("Done Training")
  return model,likelihood


def eval_gp(test_x,model,likelihood):

  print("Predicting")
  # Set into eval mode
  model.eval()
  likelihood.eval()
  
  # Make predictions
  with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
      predictions = likelihood(model(test_x))
      means = predictions.mean
      variances = predictions.variance

  return means, variances

if __name__ == "__main__":
  from datetime import datetime
  now     = datetime.now()
  seed    = int("%d%.2d%.2d%.2d%.2d"%(now.month,now.day,now.hour,now.minute,now.second))
  barcode = "%d%.2d%.2d%.2d%.2d%.2d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
  torch.random.manual_seed(seed)

  # load data
  import pickle
  d = pickle.load(open("./synthetic1_dataset_10000_points_5_dim.pickle", "rb"))
  X = d['X']
  Y = d['Y']
  n,dim = X.shape
  n_train = int(0.8*n) 
  n_test  = n - n_train
  # reduce n_train
  n_train = int(n_train/(dim+1))
  # train/test split
  train_x = X[:n_train]
  train_y = Y[:n_train]
  test_x = X[n_train:n_train+n_test]
  test_y = Y[n_train:n_train+n_test]
  test_f = test_y[:,0] # just function values
  # train gp
  num_epochs = 400
  lr_hypers = 0.05
  model,likelihood = train_gp(train_x,train_y,num_epochs=num_epochs,lr_hypers=lr_hypers,verbose=True)
  # eval gp
  means,variances = eval_gp(test_x,model,likelihood)
  means = means[:,0] # just function values
  variances= variances[:,0] # just function values
  # compute MSE
  test_mse = torch.mean((test_f-means)**2)
  # compute mean negative predictive density
  test_nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_f).mean()
  print(f"At {n_test} testing points, MSE: {test_mse:.4e}, nll: {test_nll:.4e}.")

  # file name
  data_filename = f"./output/data_ExactGradGP_ne_{num_epochs}_{barcode}.pickle"
  # dump the data
  outdata = {}
  outdata['test_mse']   = test_mse
  outdata['test_nll']   = test_nll
  outdata['mode']       = "ExactGradGP"
  outdata['dim']        = dim
  outdata['M']          = n_train
  outdata['num_epochs'] = num_epochs
  outdata['lr_hypers'] = lr_hypers
  data_filename
  pickle.dump(outdata,open(data_filename,"wb"))
  print(f"Dropped file: {data_filename}")
