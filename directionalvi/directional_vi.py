import math
import numpy as np
import torch
import gpytorch
import tqdm
import random
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import sys
#sys.path.append("../directionalvi")
sys.path.append("utils")

from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad #.RBFKernelDirectionalGrad
from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy #.DirectionalGradVariationalStrategy
from CiqDirectionalGradVariationalStrategy import CiqDirectionalGradVariationalStrategy #.DirectionalGradVariationalStrategy
try: # import wandb if watch model on weights&biases
  import wandb
except:
  pass

"""Future Upgrades
- don't include function values in every training iteration... be truly stochastic.
"""

class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self,inducing_points,inducing_directions,dim,**kwargs):

        self.num_inducing   = len(inducing_points)
        self.num_directions = int(len(inducing_directions)/self.num_inducing) # num directions per point
        num_directional_derivs = self.num_directions*self.num_inducing

        # variational distribution q(u,g)
        # variational_distribution = gpytorch.variational.DeltaVariationalDistribution(
        #     num_inducing + num_directional_derivs)
        if "variational_distribution" in kwargs and kwargs["variational_distribution"] == "NGD":
          variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            self.num_inducing + num_directional_derivs)
        else:
          variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            self.num_inducing + num_directional_derivs)

        # variational strategy q(f)
        if "variational_strategy" in kwargs and kwargs["variational_strategy"] == "CIQ":
          variational_strategy = CiqDirectionalGradVariationalStrategy(self,
            inducing_points, inducing_directions,variational_distribution, learn_inducing_locations=True)
        else:
          variational_strategy = DirectionalGradVariationalStrategy(self,
            inducing_points,inducing_directions,variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)

        # set the mean and covariance
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernelDirectionalGrad())

    def forward(self, x, **params):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x, **params)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def select_cols_of_y(y_batch,minibatch_dim,dim):
  """
  randomly select columns of y to train on, but always select 
  function values as part of the batch. Otherwise we have
  to keep track of whether we passed in function values or not
  when computing the kernel.

  input
  y_batch: 2D-torch tensor
  minibatch_dim: int, total number of derivative columns of y to sample
  dim: int, problem dimension
  """
  # randomly select columns of y to train on
  idx_y   = random.sample(range(1,dim+1),minibatch_dim) # ensures unique entries
  idx_y  += [0] # append 0 to the list for function values
  idx_y.sort()
  y_batch = y_batch[:,idx_y]

  # dont pass a direction if we load function values
  E_canonical = torch.eye(dim).to(y_batch.device)
  derivative_directions = E_canonical[np.array(idx_y[1:])-1]

  return y_batch,derivative_directions


def train_gp(train_dataset,num_inducing=128,
  num_directions=1,minibatch_size=1,minibatch_dim =1,num_epochs=1,
  learning_rate_hypers=0.01,learning_rate_ngd=0.1,
  inducing_data_initialization=True,
  use_ngd=False,
  use_ciq=False,
  lr_sched=None,
  num_contour_quadrature=15,
  watch_model=False,
  **args):
  """Train a Derivative GP with the Directional Derivative
  Variational Inference method

  train_dataset: torch Dataset
  num_inducing: int, number of inducing points
  num_directions: int, number of inducing directions (per inducing point)
  minbatch_size: int, number of data points in a minibatch
  minibatch_dim: int, number of derivative per point in minibatch training
                 WARNING: This must equal num_directions until we complete
                 the PR in GpyTorch.
  num_epochs: int, number of epochs
  inducing_data_initialization: initialize the inducing points as a set of 
      data points. If False, the inducing points are generated on the unit cube
      uniformly, U[0,1]^d.
  learning_rate_hypers, float: initial learning rate for the hyper optimizer
  learning_rate_ngd, float: initial learning rate for the variational optimizer
  use_ngd, bool: use NGD
  use_ciq, bool: use CIQ
  lr_sched, function handle: used in the torch LambdaLR learning rate scheduler. At
      each iteration the initial learning rate is multiplied by the result of 
      this function. The function input is the epoch, i.e. lr_sched(epoch). 
      The function should return a single number. If lr_sched is left as None, 
      the learning rate will be held constant.
  """
  assert num_directions == minibatch_dim

  # set up the data loader
  train_loader  = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
  dim = len(train_dataset[0][0])
  n_samples = len(train_dataset)
  num_data = (dim+1)*n_samples

  if inducing_data_initialization is True:
    # initialize inducing points and directions from data
    inducing_points = torch.zeros(num_inducing,dim)
    # canonical directions
    inducing_directions = torch.eye(dim)[:num_directions] 
    inducing_directions = inducing_directions.repeat(num_inducing,1)
    for ii in range(num_inducing):
      inducing_points[ii] = train_dataset[ii][0]
      #inducing_directions[ii*num_directions] = train_dataset[ii][1][1:] # gradient
  else:
    # random points on the unit cube
    inducing_points     = torch.rand(num_inducing, dim)
    #inducing_directions = torch.rand(num_inducing*num_directions,dim)
    #inducing_directions = (inducing_directions.T/torch.norm(inducing_directions,dim=1)).T
    inducing_directions = torch.eye(dim)[:num_directions] # canonical directions
    inducing_directions = inducing_directions.repeat(num_inducing,1)
  if torch.cuda.is_available():
    inducing_points = inducing_points.cuda()
    inducing_directions = inducing_directions.cuda()


  # initialize model
  if use_ciq:
    gpytorch.settings.num_contour_quadrature(num_contour_quadrature)
    model = GPModel(inducing_points,inducing_directions,dim, variational_distribution="NGD",variational_strategy="CIQ")
  elif use_ngd:
    model = GPModel(inducing_points,inducing_directions,dim, variational_distribution="NGD")
  else:
    model = GPModel(inducing_points,inducing_directions,dim)
  likelihood = gpytorch.likelihoods.GaussianLikelihood()
  if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()
  if watch_model:
    wandb.watch(model)
  # training mode
  model.train()
  likelihood.train()

  # optimizers
  if use_ngd or use_ciq:
    variational_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=num_data, lr=learning_rate_ngd)
    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate_hypers)
  else:
    variational_optimizer = torch.optim.Adam([
        {'params': model.variational_parameters()},
    ], lr=learning_rate_hypers)
    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate_hypers)
  
  # learning rate scheduler
  #lambda1 = lambda epoch: 1.0/(1 + epoch)
  if lr_sched is None:
    lr_sched = lambda epoch: 1.0
  hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(hyperparameter_optimizer, lr_lambda=lr_sched)
  variational_scheduler = torch.optim.lr_scheduler.LambdaLR(variational_optimizer, lr_lambda=lr_sched)

  # mll
  mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)

  # train
  print_loss=True # if print loss every 100 steps
  if "tqdm" in args and args["tqdm"]:
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
  else:
    epochs_iter = range(num_epochs)

  for i in epochs_iter:
    # iterator for minibatches
    if "tqdm" in args and args["tqdm"]:
      print_loss=False # don't print loss every 100 steps if use tqdm
      minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    else:
      minibatch_iter = train_loader
    # loop through minibatches
    mini_steps = 0
    for x_batch, y_batch in minibatch_iter:
      if torch.cuda.is_available():
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

      # select random columns of y_batch to train on
      y_batch,derivative_directions = select_cols_of_y(y_batch,minibatch_dim,dim)
      kwargs = {}
      # repeat the derivative directions for each point in x_batch
      kwargs['derivative_directions'] = derivative_directions.repeat(y_batch.size(0),1)

      # pass in interleaved data... so kernel should also interleave
      y_batch = y_batch.reshape(torch.numel(y_batch))

      variational_optimizer.zero_grad()
      hyperparameter_optimizer.zero_grad()
      output = model(x_batch,**kwargs)
      loss = -mll(output, y_batch)
      if watch_model:
        wandb.log({"loss": loss.item()})
      if "tqdm" in args and args["tqdm"]:
        epochs_iter.set_postfix(loss=loss.item())     
      loss.backward()
      # step optimizers and learning rate schedulers
      variational_optimizer.step()
      variational_scheduler.step()
      hyperparameter_optimizer.step()
      hyperparameter_scheduler.step()

      # print the loss
      if mini_steps % 10 == 0 and print_loss:
        print(f"Epoch: {i}; Step: {mini_steps}, loss: {loss.item()}")

      mini_steps +=1
      sys.stdout.flush()
     
  if print_loss:
    print(f"Done! loss: {loss.item()}")

  print("\nDone Training!")
  return model,likelihood


def eval_gp(test_dataset,model,likelihood,num_directions=1,minibatch_size=1,minibatch_dim =1):
  
  assert num_directions == minibatch_dim

  dim = len(test_dataset[0][0])
  n_test = len(test_dataset)
  test_loader = DataLoader(test_dataset, batch_size=minibatch_size, shuffle=False)
  
  model.eval()
  likelihood.eval()
  
  kwargs = {}
  means = torch.tensor([0.])
  variances = torch.tensor([0.])
  with torch.no_grad():
    for x_batch, y_batch in test_loader:
      if torch.cuda.is_available():
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
      # redo derivative directions b/c batch size is not consistent
      derivative_directions = torch.eye(dim)[:num_directions]
      derivative_directions = derivative_directions.repeat(len(x_batch),1)
      kwargs['derivative_directions'] = derivative_directions
      # predict
      preds = model(x_batch,**kwargs)
      means = torch.cat([means, preds.mean.cpu()])
      variances = torch.cat([variances, preds.variance.cpu()])

  means = means[1:]
  variances = variances[1:]

  print("Done Testing!")

  return means, variances

