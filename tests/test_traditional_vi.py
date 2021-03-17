from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
from directionalvi.utils.metrics import MSE
import testfun


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

# setups
n  = 600
n_test = 1000
dim = 2
num_inducing = 20
batch_size = int(n/2)
num_epochs = 1000

# seed
torch.random.manual_seed(0)

# trainig data
train_x = torch.rand(n,dim)
# f(x) = sin(2pi(x**2+y**2)), df/dx = cos(2pi(x**2+y**2))4pi*x, df/dy = cos(2pi(x**2+y**2))4pi*y
train_y = testfun.f(train_x, deriv=False)
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# testing data
test_x = torch.rand(n_test,dim)
test_y = testfun.f(test_x, deriv=False)
test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=n_test, shuffle=False)


print("\n\n---Standard SVGP---")
print(f"Start training with {n} trainig data of dim {dim}")
print(f"VI setups: {num_inducing} inducing points")

# setup model
inducing_points = train_x[:num_inducing, :]
# inducing_points = torch.rand(num_inducing,dim)
model = GPModel(inducing_points=inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

# model training
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},], lr=0.01)

# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        epochs_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

# model testing
model.eval()
likelihood.eval()

# for param in model.parameters():
#   if param.requires_grad:
#     print(param.data)

means = torch.tensor([0.])
variances = torch.tensor([0.])
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = model(x_batch)
        means = torch.cat([means, preds.mean.cpu()])
        variances = torch.cat([variances, preds.variance.cpu()])
means = means[1:]
variances = variances[1:]

# compute MSE
test_mse = MSE(test_y,means)
# compute mean negative predictive density
test_nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_y).mean()
print(f"At {n_test} testing points, MSE: {test_mse:.4e}, nll: {test_nll:.4e}")


# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(12,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(train_x[:,0],train_x[:,1],train_y, color='k')
# ax.scatter(train_x[:,0],train_x[:,1],means, color='b')
# plt.title("f(x,y) variational fit; actual curve is black, variational is blue")
# plt.show()