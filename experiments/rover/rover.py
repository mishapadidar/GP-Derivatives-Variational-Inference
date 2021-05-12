import numpy as np
from finite_difference import fdiff_jac
def rover_obj(u):
  """
  The rover problem:
  The goal is to learn a controller to drive a rover through four
  waypoints. 
  state: 4dim position, velocity
  control: 2dim x,y forces

  input:
  u: length 2T array, open-loop controller
  return:
  cost: float, cost associated with the controller
  """
  assert len(u) == 200

  m   = 5 # mass
  h   = 0.1 #deltat
  T   = 100 # number of steps
  eta = 1.0 # friction coeff
  q1   = 1e0  # penalty on missing waypoint
  q2   = 1e-4 # penalty on control

  # state, control
  dim_s = 4 
  dim_c = 2

  # dynamics
  A = np.array([[1,0,h,0],[0,1,0,h],[0,0,(1-eta*h/m),0],[0,0,0,(1-eta*h/m)]])
  B = np.array([[0,0],[0,0],[h/m,0],[0,h/m]])
  # waypoints
  W = np.array([[8,15,3,-4],[16,7,6,-4],[16,12,-6,-4],[0,0,0,0]])
  way_times = (np.array([.1,.4,.7,1])*T - 1).astype(int)
  
  # state control (time is a row)
  x = np.zeros((T,dim_s))
  
  # reshape the control
  u = np.reshape(u,(T,dim_c))

  # initial condition
  x[0] = [5,20,0,0]

  # dynamics
  # x_{t+1}  = Ax_t + Bu_t for t=0,...,T-1
  for t in range(0,T-1):
    x[t+1] = A @ x[t] + B @ u[t]
  
  # compute cost
  cost = q1*np.sum((x[way_times] - W)**2) + q2*np.sum(u**2)

  return cost

def rover_grad(u):
  assert len(u) == 200
  """finite difference gradient"""
  return fdiff_jac(rover_obj,u,h=1e-6)


if __name__=="__main__":
  u = np.ones(200)
  print(rover_obj(u))
  grad = rover_grad(u)
  u = u - 1e0*rover_grad(u)
  print(rover_obj(u))
  u = u - 1e0*rover_grad(u)
  print(rover_obj(u))
