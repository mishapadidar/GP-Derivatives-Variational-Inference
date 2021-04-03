import matplotlib.pyplot as plt
import pickle
import numpy as np
import glob

# read the data
dsvgp_files = glob.glob("./output/data_synthetic1_DSVGP*.pickle")
svgp_files = glob.glob("./output/data_synthetic1_SVGP*.pickle")

M_size = [[],[]] # matrix size
NLL =[[],[]]
for ff in dsvgp_files:
  # load
  d = pickle.load(open(ff, "rb"))  
  ni = d['num_inducing']
  nd = d['num_directions']
  nll = d['test_nll']
  NLL[0].append(nll.item())
  M_size[0].append(ni*(nd+1))
for ff in svgp_files:
  # load
  d = pickle.load(open(ff, "rb"))  
  ni = d['num_inducing']
  nd = d['num_directions']
  nll = d['test_nll']
  NLL[1].append(nll.item())
  M_size[1].append(ni)

print(M_size)
print(NLL)
plt.plot(M_size[0],NLL[0],label='dsvgp; ni=512')
plt.plot(M_size[1],NLL[1],label='svgp')
plt.yscale("symlog", linthreshy=0.01)
plt.title("NLL vs Inducing Matrix size")
plt.ylabel("NLL")
plt.xlabel("Inducing Matrix Size")
plt.legend()
plt.show()

