
from torch import is_tensor
from torch.utils.data import Dataset
from torch.quasirandom import SobolEngine
from rescale import from_unit_cube

class synthetic_dataset(Dataset):
    """A synthetic dataset that generates data when called from. 
    """
    def __init__(self, f,lb,ub,n_points,dim):
      """
      Args:
          f (function handle): Returns a function value and gradient eval
          lb,ub (1D tensors): lower and upper bounds on domain of f
          n_points (int): number of data points
      """
      self.f = f
      self.lb = lb
      self.ub = ub
      self.n_points = n_points
      self.dim = dim
      self.sobol = SobolEngine(dim,scramble=True)

    def __len__(self):
      return self.n_points

    def __getitem__(self, idx):
      if is_tensor(idx):
        idx = idx.tolist()
      # reset the sobol sequence
      self.sobol.reset()
      # fast forward to the desired index
      self.sobol.fast_forward(idx-1)
      # generate a point
      x = self.sobol.draw().flatten()
      # map from unit cube
      x = x * (self.ub - self.lb) + self.lb
      # evaluate it
      fx = self.f(x)
      # return a tuple of tensors
      sample = (x,fx)
      return sample
