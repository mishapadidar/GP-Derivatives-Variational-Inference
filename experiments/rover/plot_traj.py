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

num_curves = 4
means  = np.zeros((num_curves,2000))
std    = np.zeros((num_curves,2000))
n_type = np.zeros(num_curves)
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
    std[0] += fXmin**2
    n_type[0] += 1
  if d['mode'] == "Vanilla":
    means[1] += fXmin
    std[1] += fXmin**2
    n_type[1] += 1
  if d['mode'] == "Random Search":
    means[2] += fXmin
    std[2] += fXmin**2
    n_type[2] += 1
  if d['mode'] == "BO-LCB":
    means[3] += fXmin
    std[3] += fXmin**2
    n_type[3] += 1

  #plt.plot(fXmin,linewidth=5,markersize=12,color=colors[ii],label=label)

means = np.diag(1/n_type) @ means
std = np.sqrt(np.diag(1/n_type)@ std - means**2)
plt.plot(means[0],linewidth=3,markersize=12,label="TuRBO-DPPGPR1")
plt.fill_between(np.arange(0,2000),means[0]-std[0], means[0]+std[0],alpha=0.7)
plt.plot(means[1],linewidth=3,markersize=12,label="TuRBO")
plt.fill_between(np.arange(0,2000),means[1]-std[1], means[1]+std[1],alpha=0.7)
plt.plot(means[2],linewidth=3,markersize=12,label="Random Search")
plt.fill_between(np.arange(0,2000),means[2]-std[2], means[2]+std[2],alpha=0.7)
plt.plot(means[3],linewidth=3,markersize=12,label="BO-LCB")
plt.fill_between(np.arange(0,2000),means[3]-std[3], means[3]+std[3],alpha=0.7)
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



