from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import math
import time
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
sys.path.append("../directionalvi/utils")
from directionalvi.utils.metrics import MSE


class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp(train_dataset,num_inducing=128,minibatch_size=1,num_epochs=1,**args):
    if torch.cuda.is_available():
        train_dataset = train_dataset.cuda()
    train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
    # setup model
    inducing_points = train_x[:num_inducing, :]
    # inducing_points = torch.rand(num_inducing,dim)
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},], lr=0.01)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
    
    if "tqdm" in args and args["tqdm"]:
        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    else:
        epochs_iter = range(num_epochs)
        
    for i in epochs_iter:
        if "tqdm" in args and args["tqdm"]:
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        else:
            minibatch_iter = train_loader
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            #print(loss.item())
            #print(loss.shape)
            if "tqdm" in args and args["tqdm"]:
                epochs_iter.set_postfix(loss=loss.item())           
            loss.backward()
            optimizer.step()
        if i % 100 == 0:
            print(f"Training epoch {i}, loss: {loss.item()}")
    print(f"Training epoch {i}, loss: {loss.item()}")

    print("\nDone Training!")
    return model, likelihood

def eval_gp(test_dataset,model,likelihood, num_inducing=128,minibatch_size=1):
  
    dim = len(test_dataset[0][0])
    n_test = len(test_dataset)
    if torch.cuda.is_available():
        test_dataset = test_dataset.cuda()
    test_loader = DataLoader(test_dataset, batch_size=minibatch_size, shuffle=False)
    
    model.eval()
    likelihood.eval()
    
    means = torch.tensor([0.])
    variances = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
            variances = torch.cat([variances, preds.variance.cpu()])
    means = means[1:]
    variances = variances[1:]

    return means, variances
