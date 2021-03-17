def MSE(Y,Z):
  """Compute the MSE.
  Y: torch tensor, function values
  Z: torch tensor, predicted function values 
  """
  return ((Y-Z)**2).mean()

from torch.distributions import Normal
def nll(means,variances):
  """Compute negative log likelihood
  means, variances = model(test_x) 
  """
  nll = -Normal(means, variances.sqrt()).log_prob(test_y).mean() 
  return nll
