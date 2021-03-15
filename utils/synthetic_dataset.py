
import torch
from torch.utils.data import Dataset

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

    def __len__(self):
      return self.n_points

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
        idx = idx.tolist()
      # generate a point
      x = torch.rand(self.dim)*(self.ub - self.lb) + self.lb
      # evaluate it
      fx = self.f(x)
      # return a tuple of tensors
      sample = (x,fx)
      return sample
