from scipy.sparse.linalg import svds
import numpy as np


def compute_optimal_subspace_projection(G,k):
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
  P =  U[:,:k] @ np.diag(S[:k])
  return P

if __name__ == "__main__":
  import pickle
  d = pickle.load(open("synthetic1_dataset_10000_points_5_dim.pickle", "rb"))
  X = d['X'].detach().numpy()
  Y = d['Y'].detach().numpy()
  n,dim = X.shape
  f = Y[:,0].reshape(n,1)
  G = Y[:,1:]
  # compute the new gradients
  k = 2
  P = compute_optimal_subspace_projection(G,k)
  # make a reduced dataset
  Y = np.hstack((f,P))
  d = {}
  d['X'] = X
  d['Y'] = Y
  pickle.dump(d,open(f"synthetic1_dataset_10000_points_5_dim_grad_dimredux_{k}_directions.pickle","wb"))
  
