
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class csv_dataset(Dataset):
    """Reads a CSV dataset on the fly
    """
    def __init__(self,csv_file,gradients=True):
      """
      Args:
      csv_file (string): csv file name containing data
        rows have header x0,x1,...xd,f,g0,...,gd
        xi is ith x index
        f is function value
        gi is ith g index
      """
      self.df    = pd.read_csv(csv_file)
      # x indexes
      self.xidx  = np.where(['x' in ci for ci in self.df.columns])[0]
      # function values
      self.fidx  = np.where(['f' in ci for ci in self.df.columns])[0]
      # gradient indexes
      self.gidx  = np.where(['g' in ci for ci in self.df.columns])[0]
      # combined f and g indexes with f first
      self.fgidx = np.concatenate((self.fidx,self.gidx))
      # bounds for rescaling
      self.lb = torch.tensor(self.df.iloc[:,self.xidx].min(axis=0).to_numpy()).float()
      self.ub = torch.tensor(self.df.iloc[:,self.xidx].max(axis=0).to_numpy()).float()
      # gradients option
      self.gradients = gradients
 

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):
      """
      return: a tuple of torch tensors (x,y)
             x is a 2d-tensor of type float 
             y is a 1d-tensor containing function value then the gradient
      """
      if torch.is_tensor(idx):
        idx = idx.tolist()
      # get the row
      sample = self.df.iloc[idx].to_numpy()
      # return a tuple of tensors (x,[f(x),g(x)])
      x = torch.tensor(sample[self.xidx]).float() # x must be dtype float
      if self.gradients:
        y = torch.tensor(sample[self.fgidx])
      else:
        y = sample[self.fidx][0]
      # map x to unit cube, and rescale g correspondingly
      #x = (x-self.lb)/(self.ub - self.lb)
      return (x,y)

