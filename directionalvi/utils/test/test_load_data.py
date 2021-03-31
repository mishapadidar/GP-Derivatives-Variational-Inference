import os
import sys
sys.path.append("../")
from load_data import *

args = {}
args["n_train"] = 12000 
args["n_test"] = 2040
args["seed"] = 3

#cwd = os.getcwd()
#print(cwd)
#print("hi")
train, test, dim = load_helens("../../../data/MtSH.mat", **args)
print(len(train))
print(train[0])
print(len(test))
print(dim)