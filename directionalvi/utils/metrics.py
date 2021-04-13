def MSE(Y,Z):
  """Compute the MSE.
  Y: torch tensor, function values
  Z: torch tensor, predicted function values 
  """
  return ((Y-Z)**2).mean()

def MAE(Y,Z):
  """Compute the MSE.
  Y: torch tensor, function values
  Z: torch tensor, predicted function values 
  """
  return ((Y-Z).abs()).mean()

def RMSE(Y, Z):
  """Compute the MSE.
  Y: torch tensor, function values
  Z: torch tensor, predicted function values 
  """
  return ((Y-Z)**2).mean().sqrt()

def SMAE(Y, Z):
  """Compute the MSE.
  Y: torch tensor, function values
  Z: torch tensor, predicted function values 
  """
  return ((Y-Z).abs()).mean() / Y.abs().mean()