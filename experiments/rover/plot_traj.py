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

rc = {'figure.figsize':(10,5),
      'axes.facecolor':'white',
      'axes.grid' : True,
      'grid.color': '.8',
      'font.family':'Times New Roman',
      'font.size' : 20}
plt.rcParams.update(rc)
plt.figure(figsize=(10,10))

means = np.zeros((2,2000))
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
  # minimum function values
  fX = d['fX']
  fXmin = np.minimum.accumulate(fX)
  # accumulate means
  if d['mode'] == "DSVGP":
    means[0] += fXmin
    n_type[0] += 1
  if d['mode'] == "Vanilla":
    means[1] += fXmin
    n_type[1] += 1
  #plt.plot(fXmin,linewidth=5,markersize=12,color=colors[ii],label=label)

means[0] = means[0]/n_type[0]
means[1] = means[1]/n_type[1]
plt.plot(means[0],linewidth=5,markersize=12,label="TuRBO-DPPGPR1")
plt.plot(means[1],linewidth=5,markersize=12,label="TuRBO")
# plot
#sns.set()
#sns.set_style("whitegrid")
#sns.set_context("paper", font_scale=1.5)
plt.legend()
plt.title("Optimization Convergence on Rover Problem")
plt.yscale("log")
plt.ylabel("f(x)")
plt.xlabel("Evaluation")
plt.show()



