from scipy.sparse.linalg import svds
import numpy as np
import torch


def compute_optimal_subspace_projection(G,X,k):
  """Compute the optimal k-dimensional representation G
  np array, 2d: G, rows are observations
  int: k, dimension of subspace
  return:
    V: optimal directions
    S: optimal weights (eigenvalues)
    P: optimal weighted directions
  """
  # center the data
  G = G - np.mean(G,axis=0)
  # compute the k largest eigens of G
  U,S,VT = np.linalg.svd(G)
  # truncated score matrix
  #G =  U[:,:k] @ np.diag(S[:k])
  G =  G @ (VT.T)[:,:k]
  # project X as well
  X = X @ (VT.T[:,:k])
  return G,X

if __name__ == "__main__":
  import pickle
  d = pickle.load(open("synthetic1_dataset_10000_points_5_dim.pickle", "rb"))
  X = d['X'].detach().numpy()
  Y = d['Y'].detach().numpy()
  n,dim = X.shape
  f = Y[:,0].reshape(n,1)
  G = Y[:,1:]
  # compute the reduced G and X
  k = 2 # reduced dimension
  G,X = compute_optimal_subspace_projection(G,X,k)
  # make a reduced dataset
  Y = np.hstack((f,G))
  d = {}
  d['X'] = torch.tensor(X)
  d['Y'] = torch.tensor(Y)
  pickle.dump(d,open(f"synthetic1_dataset_10000_points_5_dim_grad_dimredux_{k}_directions.pickle","wb"))
  
