import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import glob
from rover import *

# read the data
data_files = glob.glob("./output/data*.pickle")
colors = pl.cm.jet(np.linspace(0,1,len(data_files)))

means = np.zeros((2,1000))
n_type = np.zeros(2)
data = []
for ii in range(len(data_files)):
  ff = data_files[ii]
  # attributes
  attrib = {}
  # load
  d = pickle.load(open(ff, "rb"))  
  if d['mode'] == 'Vanilla':
    label = "TuRBO" 
  elif d['mode'] == 'DSVGP' and d['mll_type'] == 'PLL':
    M = d['num_inducing']*(d['num_directions']+1)
    label = f"TuRBO-DPPGPR{d['num_directions']}"
  elif d['mode'] == "GD":
    label= d['mode']
  uopt = d['X'][-1]
  x0 = np.array([5,20,0,0])
  X = rover_dynamics(uopt,x0)
  plt.plot(X[:,0],X[:,1],linewidth=5,markersize=12,color=colors[ii],label=label)

# plot the waypoints
W = np.array([[8,15,3,-4],[16,7,6,-4],[16,12,-6,-4],[0,0,0,0]])
plt.scatter(W[:,0],W[:,1],color='k',s=50,label='waypoints')
# sns.set_style("whitegrid")
# sns.set_context("paper", font_scale=1.5)
plt.legend()
plt.title("Rover Path")
plt.ylabel("$x_2$")
plt.xlabel("$x_1$")
plt.show()

