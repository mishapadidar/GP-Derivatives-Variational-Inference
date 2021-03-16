def MSE(Y,Z):
  """Compute the MSE.
  Y: torch tensor, function values
  Z: torch tensor, predicted function values 
  """
  return ((Y-Z)**2).mean()
