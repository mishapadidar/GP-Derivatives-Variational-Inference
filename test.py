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
- we will pass in mini-batches of data stochastically with a fixed size of observations,
  i.e. function values plus fixed number of derivatives
- when passing in training data we should pass in X,[fX,df/dW],W where W is the set
  of canonical directions that the mini-batch of derivatives df/dW are taken in.
- upon taking block of observations we should reshape observations as a giant vector
  so that [fX,df/dW] is a giant vector. This means we should repeat the X values so that 
  the vector of observations is the same number of entries as X. 
- The kernel K_{XZ} is computes the kernel at the points X and Z in the directions U and V. 
  When X is training points the directions are W, the canonical directions that match up
  with the derivative directions of the mini-batch. The directions V are the inducing directions. 
  We must rewrite the kernel to require two sets of points and two sets of directions.
- q(f) should be a vector of same size as the number of observations. We should edit the
  base variational strategy so that we can pass in kernel directions. I beleive that we can
  do this through kwargs.
- make sure to only return the diagonal of q(f)
- One big takeaway is that their is no multitasking at all!"""

"""Future Upgrades
- don't include function values in every training iteration... be truly stochastic.
"""

class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self,num_inducing,num_directions,dim):

        self.num_directions = num_directions
        self.num_inducing   = num_inducing
        # Let's use a different set of inducing points for each latent function
        inducing_points     = torch.rand(num_inducing, dim)
        inducing_directions = torch.eye(dim)[:num_directions] # canonical directions
        num_directional_derivs = num_directions*num_inducing

        # variational distribution q(u,g)
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(
            num_inducing + num_directional_derivs)
        # variational strategy q(f)
        variational_strategy = DirectionalGradVariationalStrategy(gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True),
        num_directions
        )
        super().__init__(variational_strategy)

        # TODO: 
        # - mark the mean and covariance as batch (see example)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_directions+1]))
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernelDirectionalGrad(batch_shape=torch.Size([num_directions+1])),
          batch_shape=torch.Size([num_directions+1]))
        # set the number of directions
        self.covar_module.base_kernel.set_n_dir(num_directions)

        # this is the LMC covar... just here for testing
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_directions+1])),
        #     batch_shape=torch.Size([num_directions+1])
        # )

        # register the directions
        self.register_parameter(name="inducing_directions", parameter=torch.nn.Parameter(inducing_directions))
        # print(variational_strategy(torch.rand(10, dim)).shape)
        # quit()



    def forward(self, x, derivative_directions):

        # pass in params
        params = {}
        params['V']  =  self.inducing_directions.data
        params['num_inducing'] = self.num_inducing

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
  minibatch_dim: int, total number of columns of y to sample
  dim: int, problem dimension
  """
  # randomly select columns of y to train on
  idx_y   = random.sample(range(1,dim+1),minibatch_dim-1) # ensures unique entries
  idx_y  += [0] # append 0 to the list for function values
  idx_y.sort()
  y_batch = y_batch[:,idx_y]

  # dont pass a direction if we load function values
  E_canonical = torch.eye(dim)
  derivative_directions = E_canonical[idx_y[1:]]

  return y_batch,derivative_directions



def train_gp(train_x,train_y,num_inducing=128,
  num_directions=1,minibatch_size=1,minibatch_dim =1,num_epochs=1):
  
  dim = train_x.size(-1)

  # initialize model
  model = GPModel(num_inducing,num_directions,dim)
  likelihood = gpytorch.likelihoods.GaussianLikelihood()

  # training mode
  model.train()
  likelihood.train()

  # TODO:
  # - find out if the hyperparameters are passed through the kernel to here
  optimizer = torch.optim.Adam([
      {'params': model.parameters()},
      {'params': likelihood.parameters()},
  ], lr=0.1)

  # TODO:
  # determine if num_data=train_y.size(0) or if we should make
  # the number of data equal to the train_y.size(0)*train_y.size(1)???
  mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

  # set up the data loader
  train_dataset = TensorDataset(train_x, train_y)
  train_loader  = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)

  # train
  epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
  for i in epochs_iter:

    # iterator for minibatches
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)

    # loop through minibatches
    for x_batch, y_batch in minibatch_iter:

      # select random columns of y_batch to train on
      y_batch,derivative_directions = select_cols_of_y(y_batch,minibatch_dim,dim)

      # TODO:
      # pass in the canonical directions for the selected
      # derivatives with idx_y... i think pass in to model
      # when computing output
      optimizer.zero_grad()
      output = model(train_x,derivative_directions)
      loss   = -mll(output, train_y)
      epochs_iter.set_postfix(loss=loss.item())
      loss.backward()
      optimizer.step()

  return model,likelihood





if __name__ == "__main__":
  
  # generate training data
  n   = 100
  dim = 2
  train_x = torch.rand(n,dim)
  # f(x) = sin(x+y), df/dx = cos(x+y), df/dy = cos(x+y)
  train_y = torch.stack([torch.sin(train_x[:,0]+train_x[:,1]),
    torch.cos(train_x[:,0]+train_x[:,1]),torch.cos(train_x[:,0]+train_x[:,1])], -1)
  

  # train
  model,likelihood = train_gp(
                        train_x,
                        train_y,
                        num_inducing=20,
                        num_directions=dim,
                        minibatch_size = 7,
                        minibatch_dim = dim,
                        num_epochs = 50
                        )

  # Set into eval mode
  model.eval()
  likelihood.eval()

  