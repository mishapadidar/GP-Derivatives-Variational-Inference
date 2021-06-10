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

num_curves = 5
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
  if d['mode'] == "DSVGP" and d['num_directions']==1:
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
  if d['mode'] == "DSVGP" and d['num_directions']==2:
    means[4] += fXmin
    std[4] += fXmin**2
    n_type[4] += 1

  #plt.plot(fXmin,linewidth=5,markersize=12,color=colors[ii],label=label)

means = np.diag(1/n_type) @ means
std = np.sqrt(np.diag(1/n_type)@ std - means**2)
labels =["TuRBO-DPPGPR1","TuRBO","Random Search","BO-LCB","TuRBO-DPPGPR2"]
for ii,label in enumerate(labels):
  plt.plot(means[ii],linewidth=3,markersize=12,label=labels[ii])
  plt.fill_between(np.arange(0,2000),means[ii]-std[ii], means[ii]+std[ii],alpha=0.7)

print(means)
print(std)
print(labels)
d = {}
d['labels'] = labels
d['means'] = means
d['std'] = std
pickle.dump(d,open("rover_plot_data.pickle","wb"))
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



