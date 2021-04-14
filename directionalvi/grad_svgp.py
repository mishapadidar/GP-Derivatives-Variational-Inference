from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from GradVariationalStrategy import GradVariationalStrategy
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


class GPModel(ApproximateGP):
    def __init__(self, inducing_points,**kwargs):
        dim = inducing_points.size(1)
        if "variational_distribution" in kwargs and kwargs["variational_distribution"] == "NGD":
            variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0)*(dim+1))
        else:
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0)*(dim+1))
        if "variational_strategy" in kwargs and kwargs["variational_strategy"] == "CIQ":
            variational_strategy = gpytorch.variational.CiqVariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True)
        else:
            variational_strategy = GradVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp(train_dataset,dim,num_inducing=128,
            minibatch_size=1,
            num_epochs=1,
            use_ngd=False,
            use_ciq=False,
            learning_rate_hypers=0.01,
            learning_rate_ngd=0.1,
            lr_sched=None,
            mll_type="ELBO",
            num_contour_quadrature=15,
            watch_model=False,
            **args):
    
    print_loss=True
    train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
    n_samples = len(train_dataset)

    # setup model
    # inducing_points = train_x[:num_inducing, :]
    inducing_points = torch.rand(num_inducing,dim)
    if torch.cuda.is_available():
        inducing_points = inducing_points.cuda()

    if use_ciq:
        gpytorch.settings.num_contour_quadrature(num_contour_quadrature)
        model = GPModel(inducing_points=inducing_points,variational_distribution="NGD",variational_strategy="CIQ")
    elif use_ngd:
        model = GPModel(inducing_points=inducing_points,variational_distribution="NGD")
    else:
        model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    if watch_model:
        wandb.watch(model)

    model.train()
    likelihood.train()
    
    if "verbose" in args and args["verbose"]:
        param_total_dim = count_params(model,likelihood)

    # optimizers
    if use_ngd or use_ciq:
        variational_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=n_samples, lr=learning_rate_ngd)
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
    if lr_sched == "step_lr":
        num_batches = int(np.ceil(n_samples/minibatch_size))
        milestones = [int(num_epochs*num_batches/3), int(2*num_epochs*num_batches/3)]
        hyperparameter_scheduler = torch.optim.lr_scheduler.MultiStepLR(hyperparameter_optimizer, milestones, gamma=0.1)
        variational_scheduler = torch.optim.lr_scheduler.MultiStepLR(variational_optimizer, milestones, gamma=0.1)
    elif lr_sched is None:
        lr_sched = lambda epoch: 1.0
        hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(hyperparameter_optimizer, lr_lambda=lr_sched)
        variational_scheduler = torch.optim.lr_scheduler.LambdaLR(variational_optimizer, lr_lambda=lr_sched)
    else:
        hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(hyperparameter_optimizer, lr_lambda=lr_sched)
        variational_scheduler = torch.optim.lr_scheduler.LambdaLR(variational_optimizer, lr_lambda=lr_sched)

    # Our loss object. We're using the VariationalELBO
    if mll_type=="ELBO":
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_samples)
    elif mll_type=="PLL": 
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=n_samples)
    
    if "tqdm" in args and args["tqdm"]:
        print_loss=False # don't print loss every 100 epoch if use tqdm
        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    else:
        epochs_iter = range(num_epochs)
    
    total_step=0
    for i in epochs_iter:
        if "tqdm" in args and args["tqdm"]:
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        else:
            minibatch_iter = train_loader

        mini_steps = 0
        for x_batch, y_batch in minibatch_iter:
            if torch.cuda.is_available():
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            # pass in interleaved data
            y_batch = y_batch.reshape(torch.numel(y_batch))

            variational_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()
            if mll_type=="ELBO":
                output = model(x_batch)
            elif mll_type=="PLL": 
                output = likelihood(model(x_batch))
            loss = -mll(output, y_batch)
            if watch_model:
                wandb.log({"loss": loss.item()})
            loss.backward()
            # step optimizers and learning rate schedulers
            variational_optimizer.step()
            variational_scheduler.step()
            hyperparameter_optimizer.step()
            hyperparameter_scheduler.step()

            if "tqdm" in args and args["tqdm"]:
                epochs_iter.set_postfix(loss=loss.item())           
            
            if total_step % 25 == 0 and print_loss:
                means = output.mean[::dim+1]
                stds  = output.variance.sqrt()[::dim+1]
                nll   = -torch.distributions.Normal(means, stds).log_prob(y_batch[::dim+1]).mean()
                print(f"Epoch: {i}; total_step: {mini_steps}, loss: {loss.item()}, nll: {nll}")

            mini_steps +=1
            total_step +=1
            sys.stdout.flush()
        
    
     
    if print_loss:
        print(f"Done! loss: {loss.item()}")

    print("\nDone Training!")
    sys.stdout.flush()
    return model, likelihood

def eval_gp(test_dataset,model,likelihood, num_inducing=128,minibatch_size=1):
  
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
            means = torch.cat([means, preds.mean.cpu()])
            variances = torch.cat([variances, preds.variance.cpu()])
    means = means[1:]
    variances = variances[1:]

    return means, variances
