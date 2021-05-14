import numpy as np
from rover import rover_obj
import sys

"""
.. module:: example_expected_improvement
  :synopsis: Example Expected Improvement
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

import logging
import os.path

import numpy as np
from poap.controller import BasicWorkerThread, ThreadController

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.optimization_problems import OptimizationProblem
from pySOT.strategy import EIStrategy
from pySOT.surrogate import GPRegressor



class Rover(OptimizationProblem):
    def __init__(self):
        self.dim = 200
        self.min = 0
        self.minimum = np.zeros(self.dim)
        self.lb = -5 * np.ones(self.dim)
        self.ub = 5 * np.ones(self.dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)
        self.X = np.zeros((0,self.dim))
        self.fX = np.zeros(0)

    def eval(self, x):
        #self.__check_input__(x)
        f =rover_obj(x)
        #print(f)
        #sys.stdout.flush()
        #self.X = np.vstack((self.X,x))
        #self.fX = np.vstack((self.fX,f))
        return f

#if not os.path.exists("./logfiles"):
#    os.makedirs("logfiles")
#if os.path.exists("./logfiles/example_simple.log"):
#    os.remove("./logfiles/example_simple.log")
#logging.basicConfig(filename="./logfiles/example_simple.log", level=logging.INFO)

myRover = Rover()
num_threads = 4
n_init = 2*myRover.dim + 1
max_evals = 1000
gp = GPRegressor(dim=myRover.dim, lb=myRover.lb, ub=myRover.ub)
slhd = SymmetricLatinHypercube(dim=myRover.dim, num_pts=n_init)

# Create a strategy and a controller
controller = ThreadController()
controller.strategy = EIStrategy(
    max_evals=max_evals, opt_prob=myRover, exp_design=slhd, surrogate=gp, asynchronous=True
)

print("Number of threads: {}".format(num_threads))
print("Maximum number of evaluations: {}".format(max_evals))
print("Strategy: {}".format(controller.strategy.__class__.__name__))
print("Experimental design: {}".format(slhd.__class__.__name__))
print("Surrogate: {}".format(gp.__class__.__name__))

# Launch the threads and give them access to the objective function
worker = BasicWorkerThread(controller, myRover.eval)
controller.launch_worker(worker)

# Run the optimization strategy
result = controller.run()
    
result = controller.run()
fX = np.array(
    [o.value for o in controller.fevals if o.value is not None])

print("Best value found: {0}".format(result.value))
print(
    "Best solution found: {0}\n".format(
        np.array_str(result.params[0], max_line_width=np.inf, precision=5, suppress_small=True)
    )
)

d ={}
d['fX'] = fX
d['mode'] = "BO-EI"
outfilename = f"./output/data_rover_BO_{max_evals}_evals.pickle"
pickle.dump(d,open(outfilename,"wb"))



