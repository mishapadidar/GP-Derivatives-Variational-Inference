import math
import numpy as np
import torch
import gpytorch
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad



class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self,num_inducing,num_directions,dim):

        self.num_directions = num_directions
        self.num_inducing = num_inducing
        # Let's use a different set of inducing points for each latent function
        inducing_points     = torch.rand(num_inducing, dim)
        inducing_directions = torch.eye(dim)[:num_directions] # canonical directions
        num_directional_derivs = num_directions*num_inducing

        # variational distribution q(u,g)
        # use batch so we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(
            num_inducing+num_directional_derivs,batch_shape=torch.Size([inducing_directions+1]))
        # variational strategy q(f)
        # TODO: 
        # - wrap with multioutput (see example)
        variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)

        # TODO: 
        # - mark the mean and covariance as batch (see example)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernelDirectionalGrad(num_directions))
        # set the number of directions
        self.covar_module.base_kernel.set_n_dir(num_directions)

        # register the directions
        self.register_parameter(name="inducing_directions", parameter=torch.nn.Parameter(inducing_directions))



    def forward(self, x):

        # pass in params
        params = {}
        params['V']  =  self.inducing_directions.data
        params['num_inducing'] = self.num_inducing

        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x, **params)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def sample_contiguous_columns(n_max,n_cols,n_min=0):
  """
  n_max: maximum index;
  n_cols: number of columns to select
  n_min: minimum index;
  return: list of indexes of contiguous selected columns
  """
  if n_min == n_max:
    return [n_min]
  idx_start = np.random.randint(n_min,n_max)
  cols = list(range(idx_start,min([idx_start+n_cols,n_max+1])))
  if idx_start+n_cols >n_max+1:
    cols += list(range(n_min,n_min+(idx_start+n_cols)%(n_max+1)))
  return cols


def train_gp(train_x,train_y,num_inducing=128,
  num_directions=1,batch_size=1024,batch_n_partials =1,num_epochs=1):
  
  dim = train_x.size(-1)

  # Multitask GP model
  model = MultitaskGPModel(num_inducing,num_directions,dim)
  # gaussian likelihood 
  likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_directions+1)

  model.train()
  likelihood.train()

  # TODO:
  # - find out if the hyperparameters are passed through the kernel to here
  optimizer = torch.optim.Adam([
      {'params': model.parameters()},
      {'params': likelihood.parameters()},
  ], lr=0.1)

  # loss; 
  mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

  # set up the data loader
  train_dataset = TensorDataset(train_x, train_y)
  train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
  for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:

      # randomly select continguous columns of partial derivs
      # idx_deriv_contig = sample_contiguous_columns(dim,batch_n_partials,n_min=1)
      # # prepend zero so we get function values as well
      # idx_contig = [0] + idx_deriv_contig
      # y_batch = y_batch[:,idx_contig]

      optimizer.zero_grad()

      # TODO:
      # - pass the right output to the mll so that expected_log_prob is computed
      #   correctly. This might mean computing a different mean and kernel matrix
      #   than what model returns
      output = model(train_x)
      # evaluate loss
      loss = -mll(output, train_y)
      epochs_iter.set_postfix(loss=loss.item())
      loss.backward()
      optimizer.step()

  return model,likelihood





if __name__ == "__main__":
  
  # generate training data
  n   = 100
  dim = 2
  train_x = torch.rand(n,dim)
  # train_x = torch.linspace(0, 2*math.pi, n).reshape(n,dim)

  # [f(x) df/dx df/dy]
  # train_y = torch.stack([torch.sin(train_x[:,0] + train_x[:,1]),
  #     torch.cos(train_x[:,0] + train_x[:,1]),
  #     torch.cos(train_x[:,0] + train_x[:,1])
  # ], -1)
  train_y = torch.stack([torch.sin(train_x[:,0]),
      torch.cos(train_x[:,0]),
  ], -1)
  
  num_inducing = 20
  num_directions = dim
  num_epochs = 50

  # train
  model,likelihood = train_gp(
                        train_x,
                        train_y,
                        num_inducing=num_inducing,
                        num_directions=num_directions,
                        batch_size = 1024,
                        batch_n_partials =dim,
                        num_epochs = num_epochs
                        )

  # Set into eval mode
  model.eval()
  likelihood.eval()

  # Make predictions
  with torch.no_grad(), gpytorch.settings.fast_pred_var():
      n_test = 33
      # test_x = torch.rand(n_test,dim)
      test_x = torch.linspace(0, 2*math.pi, n_test).reshape(n_test,dim)
      predictions = likelihood(model(test_x))
      mean = predictions.mean
      lower, upper = predictions.confidence_region()
  print(mean)
  plt.plot(train_x.detach().numpy().flatten(), train_y[:, 0].detach().numpy().flatten(), 'k*')
  plt.plot(test_x.detach().numpy().flatten(), mean[:, 0].detach().numpy().flatten(), 'b')

  # Initialize plots
  # fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))
  # for task, ax in enumerate(axs):
  #     # Plot training data as black stars
  #     ax.plot(train_x.detach().numpy().flatten(), train_y[:, task].detach().numpy().flatten(), 'k*')
  #     # Predictive mean as blue line
  #     ax.plot(test_x.detach().numpy().flatten(), mean[:, task].detach().numpy().flatten(), 'b')
  #     # Shade in confidence
  #     # ax.fill_between(test_x.numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
  #     ax.set_ylim([-3, 3])
  #     # ax.legend(['Observed Data', 'Mean', 'Confidence'])
  #     ax.set_title(f'Task {task + 1}')
  # fig.tight_layout()
  plt.show()
  