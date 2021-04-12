from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import math
import time
import torch
import sys
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
from utils.count_params import count_params
try: # import wandb if watch model on weights&biases
  import wandb
except:
  pass


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, dim, num_tasks, num_latents, num_inducing):
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, num_inducing, dim)
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp(train_dataset,num_inducing=128,
            minibatch_size=1,num_epochs=1,
            learning_rate=0.01,
            use_ngd=False,
            use_ciq=False,
            lr_sched=None,
            mll_type="ELBO",
            watch_model=False,
            **args):

    # set up the data loader
    train_loader  = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
    dim = len(train_dataset[0][0])
    n_samples = len(train_dataset)
    num_data = (dim+1)*n_samples
    num_tasks = dim+1
    num_latents = num_tasks

    model = MultitaskGPModel(dim, num_tasks, num_latents, num_inducing)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    if watch_model:
        wandb.watch(model)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate)

    if "verbose" in args and args["verbose"]:
        param_total_dim = count_params(model,likelihood)

    # learning rate scheduler
    #lambda1 = lambda epoch: 1.0/(1 + epoch)
    if lr_sched == "step_lr":
        num_batches = int(np.ceil(n_samples/minibatch_size))
        milestones = [int(num_epochs*num_batches/3), int(2*num_epochs*num_batches/3)]
        parameter_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    elif lr_sched is None:
        lr_sched = lambda epoch: 1.0
        parameter_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_sched)
    else:
        parameter_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_sched)

    # mll
    if mll_type=="ELBO":
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_samples)
    elif mll_type=="PLL": 
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=n_samples)
    
    # train
    print_loss=True # if print loss every 100 steps
    if "tqdm" in args and args["tqdm"]:
        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    else:
        epochs_iter = range(num_epochs)
    
    total_step=0
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

            optimizer.zero_grad()
            #output = model(x_batch)
            output = likelihood(model(x_batch))
            loss = -mll(output, y_batch)
            if watch_model:
                wandb.log({"loss": loss.item()})
            loss.backward()
            # step optimizers and learning rate schedulers
            optimizer.step()
            parameter_scheduler.step()
            
            if "tqdm" in args and args["tqdm"]:
                epochs_iter.set_postfix(loss=loss.item())           

            if total_step % 50 == 0 and print_loss:
                means = output.mean
                stds  = output.variance.sqrt()
                nll   = -torch.distributions.Normal(means, stds).log_prob(y_batch).mean()
                print(f"Epoch: {i}; total_step: {mini_steps}, loss: {loss.item()}, nll: {nll}")

            mini_steps +=1
            total_step +=1
            
        # print the loss
        # if i % 20 == 0 and print_loss:
        #     print(f"Epoch: {i}; Step: {mini_steps}, loss: {loss.item()}")
     
    if print_loss:
        print(f"Done! loss: {loss.item()}")

    print("\nDone Training!")
    sys.stdout.flush()
    return model, likelihood




def eval_gp(test_dataset,model,likelihood,minibatch_size=1):

    dim = len(test_dataset[0][0])
    n_test = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=minibatch_size, shuffle=False)
  
    model.eval()
    likelihood.eval()
    
    means = torch.tensor([0.])
    variances = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if torch.cuda.is_available():
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            preds = likelihood(model(x_batch))
            means = torch.cat([means, preds.mean[:, 0].cpu()])
            variances = torch.cat([variances, preds.variance[:, 0].cpu()])
    means = means[1:]
    variances = variances[1:]

    return means, variances

