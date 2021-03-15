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
        # inducing_directions = torch.rand(num_inducing*num_directions,dim)
        # inducing_directions = (inducing_directions.T/torch.norm(inducing_directions,dim=1)).T
        inducing_directions = torch.eye(dim)[:num_directions] # canonical directions
        inducing_directions = inducing_directions.repeat(num_inducing,1)
        num_directional_derivs = num_directions*num_inducing

        # variational distribution q(u,g)
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(
            num_inducing + num_directional_derivs)
        # variational strategy q(f)
        variational_strategy = DirectionalGradVariationalStrategy(self, 
          inducing_points,inducing_directions, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)

        # set the mean and covariance
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernelDirectionalGrad())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad())


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



def train_gp(train_x,train_y,num_inducing=128,
  num_directions=1,minibatch_size=1,minibatch_dim =1,num_epochs=1):
  
  dim = train_x.size(-1)

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

  num_data = train_y.size(0)*train_y.size(1)
  mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)

  # set up the data loader
  train_dataset = TensorDataset(train_x, train_y)
  train_loader  = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)

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





if __name__ == "__main__":
  
  torch.random.manual_seed(0)
  # generate training data
  n   = 400
  dim = 2
  train_x = torch.rand(n,dim)
  # f(x) = sin(2pi(x**2+y**2)), df/dx = cos(2pi(x**2+y**2))4pi*x, df/dy = cos(2pi(x**2+y**2))4pi*y
  train_y = torch.stack([torch.sin(2*np.pi*(train_x[:,0]**2+train_x[:,1]**2)),
    4*np.pi*train_x[:,0]*torch.cos(2*np.pi*(train_x[:,0]**2+train_x[:,1]**2)),
    4*np.pi*train_x[:,1]*torch.cos(2*np.pi*(train_x[:,0]**2+train_x[:,1]**2))], -1)
  # train_y = torch.stack([torch.sin(2*np.pi*train_x[:,0]),2*np.pi*torch.cos(2*np.pi*train_x[:,0]),0.0*train_x[:,1]], -1)


  # generate training data
  # n   = 100
  # dim = 1
  # train_x = torch.linspace(0,2*np.pi,n).reshape(n,dim)
  # train_y = torch.stack([torch.sin(train_x[:,0]),
  #   torch.cos(train_x[:,0])], -1)
  # train_y = torch.stack([torch.sin(train_x[:,0])+torch.exp(-3*train_x[:,0]),
  #   torch.cos(train_x[:,0]) -3*torch.exp(-3*train_x[:,0])], -1)

  num_directions = dim
  # train
  model,likelihood = train_gp(
                        train_x,
                        train_y,
                        num_inducing=50,
                        num_directions=num_directions,
                        minibatch_size = int(n/2),
                        minibatch_dim = num_directions,
                        num_epochs =400
                        )

  # Set into eval mode
  model.eval()
  likelihood.eval()

  # predict
  kwargs = {}
  derivative_directions = torch.eye(dim)[:num_directions]
  derivative_directions = derivative_directions.repeat(n,1)
  kwargs['derivative_directions'] = derivative_directions
  preds   = model(train_x, **kwargs).mean.cpu()

  # import matplotlib.pyplot as plt
  # pred_f  = preds[::dim+1]
  # pred_df = preds[1::dim+1]
  # plt.plot(train_x.flatten(),train_y.flatten()[::dim+1],'r-',linewidth=2,label='true f(x)')
  # plt.plot(train_x.flatten(),pred_f.detach().numpy(),'b-',linewidth=2,label='variational f(x)')
  # plt.plot(train_x.flatten(),train_y.flatten()[1::dim+1],'r--',linewidth=2,label='true df/dx')
  # plt.plot(train_x.flatten(),pred_df.detach().numpy(),'b--',linewidth=2,label='variational df/dx')
  # plt.legend()
  # plt.tight_layout()
  # plt.title("Variational GP Predictions with Learned Derivatives")
  # plt.show()

  from mpl_toolkits.mplot3d import axes3d
  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(12,6))
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(train_x[:,0],train_x[:,1],train_y[:,0], color='k')
  ax.scatter(train_x[:,0],train_x[:,1],preds.detach().numpy()[::num_directions+1], color='b')
  plt.title("f(x,y) variational fit; actual curve is black, variational is blue")
  plt.show()
  fig = plt.figure(figsize=(12,6))
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(train_x[:,0],train_x[:,1],train_y[:,1], color='k')
  ax.scatter(train_x[:,0],train_x[:,1],preds.detach().numpy()[1::num_directions+1], color='b')
  plt.title("df/dx variational fit; actual curve is black, variational is blue")
  plt.show()
  fig = plt.figure(figsize=(12,6))
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(train_x[:,0],train_x[:,1],train_y[:,2], color='k')
  ax.scatter(train_x[:,0],train_x[:,1],preds.detach().numpy()[2::dim+1], color='b')
  plt.title("df/dy variational fit; actual curve is black, variational is blue")
  plt.show()