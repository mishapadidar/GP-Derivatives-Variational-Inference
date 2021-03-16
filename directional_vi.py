import math
import numpy as np
import torch
import gpytorch
import tqdm
import random
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad
from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy

"""Notes from Jake
[x] we will pass in mini-batches of data stochastically with a fixed size of observations,
  i.e. function values plus fixed number of derivatives
[x] when passing in training data we should pass in X,[fX,df/dW],W where W is the set
  of canonical directions that the mini-batch of derivatives df/dW are taken in.
[x] upon taking block of observations we should reshape observations as a giant vector
  so that [fX,df/dW] is a giant vector. This means we should repeat the X values so that 
  the vector of observations is the same number of entries as X. 
[x] The kernel K_{XZ} computes the kernel at the points X and Z in the directions U and V. 
  When X is training points the directions are W, the canonical directions that match up
  with the derivative directions of the mini-batch. The directions V are the inducing directions. 
  We must rewrite the kernel to require two sets of points and two sets of directions.
[] q(f) should be a vector of same size as the number of observations. We should edit the
  base variational strategy so that we can pass in kernel directions. I beleive that we can
  do this through kwargs.
[] make sure to only return the diagonal of the predictive covariance q(f)
[x] One big takeaway is that their is no multitasking at all!"""

"""Future Upgrades
- don't include function values in every training iteration... be truly stochastic.
"""

class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self,num_inducing,num_directions,dim):

        self.num_directions = num_directions # num directions per point
        self.num_inducing   = num_inducing
        # Let's use a different set of inducing points for each latent function
        inducing_points     = torch.rand(num_inducing, dim)
        inducing_directions = torch.rand(num_inducing*num_directions,dim)
        inducing_directions = (inducing_directions.T/torch.norm(inducing_directions,dim=1)).T
        # inducing_directions = torch.eye(dim)[:num_directions] # canonical directions
        # inducing_directions = inducing_directions.repeat(num_inducing,1)
        num_directional_derivs = num_directions*num_inducing


        # variational distribution q(u,g)
        # variational_distribution = gpytorch.variational.DeltaVariationalDistribution(
        #     num_inducing + num_directional_derivs)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing + num_directional_derivs)
        # variational strategy q(f)
        variational_strategy = DirectionalGradVariationalStrategy(self, 
          inducing_points,inducing_directions, variational_distribution, learn_inducing_locations=True)
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
  E_canonical = torch.eye(dim)
  derivative_directions = E_canonical[np.array(idx_y[1:])-1]

  return y_batch,derivative_directions



def train_gp(train_dataset,num_inducing=128,
  num_directions=1,minibatch_size=1,minibatch_dim =1,num_epochs=1):
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
  """
  
  # set up the data loader
  train_loader  = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
  dim = len(train_dataset[0][0])
  n_samples = len(train_dataset)

  # initialize model
  model = GPModel(num_inducing,num_directions,dim)
  likelihood = gpytorch.likelihoods.GaussianLikelihood()

  # training mode
  model.train()
  likelihood.train()

  optimizer = torch.optim.Adam([
      {'params': model.parameters()},
      {'params': likelihood.parameters()},
  ], lr=0.01)

  num_data = (dim+1)*n_samples
  mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)


  # train
  epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch",leave=False)
  for i in epochs_iter:

    # iterator for minibatches
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch",leave=False)

    # loop through minibatches
    for x_batch, y_batch in minibatch_iter:

      # select random columns of y_batch to train on
      y_batch,derivative_directions = select_cols_of_y(y_batch,minibatch_dim,dim)
      kwargs = {}
      # repeat the derivative directions for each point in x_batch
      kwargs['derivative_directions'] = derivative_directions.repeat(y_batch.size(0),1)

      # pass in interleaved data... so kernel should also interleave
      y_batch = y_batch.reshape(torch.numel(y_batch))
      # x_batch = x_batch.repeat_interleave(minibatch_dim+1,dim=0)

      optimizer.zero_grad()
      output = model(x_batch,**kwargs)
      loss   = -mll(output, y_batch)
      epochs_iter.set_postfix(loss=loss.item())
      loss.backward()
      optimizer.step()

  print("\nDone Training!")
  return model,likelihood

