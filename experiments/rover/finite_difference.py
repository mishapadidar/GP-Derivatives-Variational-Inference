import numpy as np
import multiprocessing as mp


def fdiff_jac(f,x0,h=1e-6):
  """Compute the jacobian of f with 
  central difference
  """
  h2   = h/2.0
  dim  = len(x0)
  Ep   = x0 + h2*np.eye(dim)
  Fp   = np.array([f(e) for e in Ep])
  Em   = x0 - h2*np.eye(dim)
  Fm   = np.array([f(e) for e in Em])
  jac = (Fp - Fm)/(h)
  return jac.T

def fdiff_jac_mp(f,x0,h=1e-6,n_comp=1):
  """Compute the jacobian of f with 
  central difference
  using multiprocessing for acceleration.
  """
  h2   = h/2.0
  dim  = len(x0)
  Ep   = x0 + h2*np.eye(dim)
  Em   = x0 - h2*np.eye(dim)
  with mp.Pool(n_comp) as p:
    Fp = np.array(p.map(f, Ep))
    Fm = np.array(p.map(f, Em))
  jac = (Fp - Fm)/(h)
  return jac.T

if __name__ == '__main__':
  np.random.seed(0)
  dim = 4
  A = np.random.randn(dim,dim)
  print(A)
  f = lambda x: A @ x
  x0   = np.random.randn(dim)
  t0 = time.time()
  print(fdiff_jac(f,x0))
