import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import glob

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
    label = d['mode'] + f" {d['turbo_batch_size']}"
  elif d['mode'] == 'DSVGP' and d['mll_type'] == 'PLL':
    M = d['num_inducing']*(d['num_directions']+1)
    label = "DPPGPR" + f"; M={M}" + f" {d['turbo_batch_size']}"
  elif d['mode'] == "GD":
    label= d['mode']
  # minimum function values
  fX = d['fX']
  fXmin = np.minimum.accumulate(fX)
  # accumulate means
  #if d['mode'] == "DPPGPR":
  #  means[0] += fXmin
  #  n_type[0] += 1
  #if d['mode'] == "Vanilla":
  #  means[1] += fXmin
  #  n_type[1] += 1
  plt.plot(fXmin,linewidth=5,markersize=12,color=colors[ii],label=label)

#means[0] = means[0]/n_type[0]
#means[1] = means[1]/n_type[1]
#plt.plot(means[0],linewidth=5,markersize=12,label="DPPGPR")
#plt.plot(means[1],linewidth=5,markersize=12,label="PPGPR")
# plot
rc = {'figure.figsize':(10,5),
      'axes.facecolor':'white',
      'axes.grid' : True,
      'grid.color': '.8',
      'font.family':'Times New Roman',
      'font.size' : 15}
plt.rcParams.update(rc)
#sns.set()
#sns.set_style("whitegrid")
#sns.set_context("paper", font_scale=2.0)
plt.legend()
plt.title("Convergence of TuRBO for varying Variational GP")
plt.yscale("log")
plt.ylabel("f(x)")
plt.xlabel("Evaluation")
plt.show()



