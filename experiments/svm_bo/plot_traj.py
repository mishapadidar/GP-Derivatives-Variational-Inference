import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import glob

# read the data
data_files = glob.glob("./output/data*.pickle")

data = []
for ff in data_files:
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
  label = d['mode'] + str(d['num_directions'])
  # minimum function values
  fX = d['fX']
  fXmin = np.minimum.accumulate(fX)
  plt.plot(fXmin,linewidth=5,markersize=12,label=label)

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
plt.ylabel("f(x)")
plt.xlabel("Evaluation")
plt.show()



