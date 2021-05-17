import numpy as np
from rover import rover_obj
import sys



if __name__ == '__main__':

  dim = 200
  max_evals = 10
  lb = -5 * np.ones(dim)
  ub = 5 * np.ones(dim)
  batch_size = 5
  num_epochs = 30

  from datetime import datetime
  now     = datetime.now()
  seed    = int("%d%.2d%.2d%.2d%.2d"%(now.month,now.day,now.hour,now.minute,now.second))
  barcode = "%d%.2d%.2d%.2d%.2d%.2d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
  np.random.seed(seed)

  X = np.random.uniform(lb,ub,(max_evals,dim))
  fX = [rover_obj(x) for x in X]

  d ={}
  d['X']  = X
  d['fX'] = fX
  d['mode'] = "Random Search"
  outfilename = f"./output/data_rover_Random_Search_{max_evals}_evals_{barcode}.pickle"
  import pickle
  pickle.dump(d,open(outfilename,"wb"))



