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

data = []
for ii in range(len(data_files)):
  ff = data_files[ii]
  # attributes
  attrib = {}
  # load
  d = pickle.load(open(ff, "rb"))  
  if d['mode'] == 'SVGP':
    d['num_directions']= 0
  # add an indicator attribute for plotting
  if d['mode'] == 'SVGP' and d['mll_type'] == 'PLL':
    d['mode'] = "PPGPR"
  elif d['mode'] == 'DSVGP' and d['mll_type'] == 'PLL':
    d['mode'] = "DPPGPR"
  M = d['num_inducing']*(d['num_directions']+1)
  label = d['mode'] + f"; M={M}" + f" {d['turbo_batch_size']}"
  # minimum function values
  fX = d['fX']
  fXmin = np.minimum.accumulate(fX)
  plt.plot(fXmin,linewidth=5,markersize=12,color=colors[ii],label=label)
  #plt.scatter(np.arange(0,len(fX)),fX,color=colors[ii])

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



