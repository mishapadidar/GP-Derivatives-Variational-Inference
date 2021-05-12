import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import glob
import os

# read the data
def main(dataset, exp_name, data_type='fX'):
  assert data_type == 'fX' or data_type == 'train_acc_list' or data_type == 'test_acc_list'
  data_files = glob.glob(f"./results/{dataset}*exp{exp_name}.pickle")
  print(data_files)
  # colors = pl.cm.jet(np.linspace(0,1,len(data_files)))
  colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  data = []
  for ii in range(len(data_files)):
    ff = data_files[ii]
    # attributes
    attrib = {}
    # load
    d = pickle.load(open(ff, "rb"))
    # d['turbo_batch_size'] = 20  
    # add an indicator attribute for plotting
    
    if d['model'] == 'SVGP' and d['mll_type'] == 'PLL':
      d['model'] = "PPGPR"
    elif d['model'] == 'DSVGP' and d['mll_type'] == 'PLL':
      d['model'] = "DPPGPR"
    
    if d['model'] == "SGD" or d['model'] == "TURBO":
      label = d['model']
    elif d['model'].startswith("D"):
      M = d['num_inducing']*(d['num_directions']+1)
      label = d['model'] + f"; M={M}; p={d['num_directions']}"
    else:
      M = d['num_inducing']
      label = d['model'] + f"; M={M}"
    # minimum function values
    fX = d[data_type]
    fXmin = np.minimum.accumulate(fX) if data_type == 'fX' else np.maximum.accumulate(fX)
    plt.plot(fXmin,linewidth=3,markersize=12,color=colors[ii],label=label)


  # plot
  rc = {'figure.figsize':(12,6),
        'axes.facecolor':'white',
        'axes.grid' : True,
        'grid.color': '.8',
        'font.family':'Times New Roman',
        'font.size' : 12}
  plt.rcParams.update(rc)
  #sns.set()
  #sns.set_style("whitegrid")
  #sns.set_context("paper", font_scale=2.0)
  plt.legend()
  plt.title("Convergence of TuRBO for varying Variational GP")
  if data_type == 'fX':
    ylabel = 'f(x)'
    plt.yscale("log")
  elif data_type == 'test_acc_list':
    ylabel = 'Test accuracy'
  elif data_type == 'train_acc_list':
    ylabel = 'Train accuracy'
    
  plt.ylabel(ylabel)
  plt.xlabel("Evaluation")
  plt.show()
  figurename = f"TuRBO_exp{exp_name}_{data_type}.pdf"
  figurepath = os.path.abspath(__file__ + "/../plots/" + figurename)
  plt.savefig(figurepath)
  print("Figure saved:", figurepath)

if __name__ == "__main__":
  dataset="PubMed"
  exp_name="M4"
  # for data_type in ['fX', 'train_acc_list', 'test_acc_list']:
  data_type='test_acc_list'
  main(dataset, exp_name, data_type)

  

