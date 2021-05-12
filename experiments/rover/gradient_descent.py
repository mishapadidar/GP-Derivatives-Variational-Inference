import numpy as np

def gradient_descent(Loss,grad,x0,max_iter=1000,gtol=1e-3):
  # inital guess
  x_k = np.copy(x0)
  # initialize step size
  mu_k   = 1e-2
  # minimum step size
  mu_min = 1e-10
  # compute gradient
  g_k    = grad(x_k)
  # compute function value
  f_k    = Loss(x_k)

  # storage
  dim = len(x0)
  X = np.zeros((1,dim))
  X[0] = np.copy(x0)

  # stop when gradient is flat (within tolerance)
  nn = 0
  while np.linalg.norm(g_k) > gtol and nn < max_iter:
    if nn%100 == 0:
      print(nn,f_k)
    # double the step size to counter backtracking
    mu_k = 2*mu_k;
    
    # compute step 
    x_kp1 = x_k -mu_k*g_k;
    f_kp1 = Loss(x_kp1);
    
    # backtracking to find step size
    while f_kp1 >= f_k:
      # half our step size
      mu_k = mu_k /2 ;
      # take step
      x_kp1 = x_k -mu_k*g_k;
      # f_kp1
      f_kp1 = Loss(x_kp1);

      # break if mu is too small
      if mu_k <= mu_min:
        print('ERROR: mu too small.')
        return x_k

    # reset for next iteration
    x_k   = np.copy(x_kp1)
    f_k   = f_kp1;
    
    # compute gradient
    g_k  = grad(x_k);

    # update iteration counter
    nn += 1
    X = np.copy(np.vstack((X,x_k)))

  return x_k,X


if __name__ == '__main__':
  f = lambda x: x @ x
  g = lambda x: 2*x
  dim = 2
  x0 = 10*np.random.randn(dim)
  xopt,X = gradient_descent(f,g,x0,max_iter=200,gtol=1e-7)
  print(xopt)
  print(X)