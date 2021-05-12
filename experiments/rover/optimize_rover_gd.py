import numpy as np 
from rover import *
from gradient_descent import gradient_descent
import pickle

# generate a starting point
dim = 200
x0 = np.random.uniform(-5,5,size=dim)
max_iter = 1000
gtol = 1e-7
# optimize
xopt,X = gradient_descent(rover_obj,rover_grad,x0,max_iter=max_iter,gtol=gtol)
fX = np.array([rover_obj(x) for x in X])
print(fX[-1])
# save data
d = {}
d['X'] = X
d['fX'] = fX
d['mode'] = "GD"
outfilename = f"./output/data_rover_GD_{max_iter}_iter.pickle"
pickle.dump(d,open(outfilename,"wb"))
