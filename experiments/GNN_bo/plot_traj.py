import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import glob
import os

# read the data
def plot_single(dataset, exp_name, data_type='fX'):
  assert data_type == 'fX' or data_type == 'train_acc_list' or data_type == 'test_acc_list'
  data_files = glob.glob(f"./results/{dataset}*exp{exp_name}.pickle")
  data_files = sorted(data_files)
  # colors = pl.cm.jet(np.linspace(0,1,len(data_files)))
  colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  
  plt.clf()
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
      label = d['model'] + f"{d['num_directions']}"
    else:
      M = d['num_inducing']
      label = d['model'] 
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
    plt.ylim( (10**-2,10**2) )
    plt.yscale("log")
  elif data_type == 'test_acc_list':
    ylabel = 'Test accuracy'
  elif data_type == 'train_acc_list':
    plt.ylim((0.28, 1.05))
    ylabel = 'Train accuracy'
    
  plt.ylabel(ylabel)
  plt.xlabel("Evaluation")
  plt.show()
  figurename = f"TuRBO_exp{exp_name}_{data_type}.pdf"
  figurepath = os.path.abspath(__file__ + "/../plots/" + figurename)
  plt.savefig(figurepath)
  print("Figure saved:", figurepath)


def plot_average(dataset, methods_list, data_type='fX', deleted_methods=None):
  plt.clf()
  plt.cla()

  assert data_type == 'fX' or data_type == 'train_acc_list' or data_type == 'test_acc_list'

  # sort to fix color for each method
  colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  style_dict = {"SGD": ["Gradient Descent", "dotted", '#1f77b4'],
                "TURBO": ["TuRBO", "dashed", '#ff7f0e'],
                "DSVGP1": ["TuRBO-DPPGPR1", "solid", '#2ca02c'],
                "DSVGP2": ["TuRBO-DPPGPR2", "solid", '#d62728'],
                "SVGP": ["TuRBO-PPGPR", "solid", '#9467bd'],
                "BO": ["BO", "solid", '#8c564b'],
                "random": ["Random", "solid", "#e377c2"] }

  # collect data for each method
  data_files_dict = {}
  for i, method in enumerate(methods_list):
    data_files_dict[method] = glob.glob(f"./results/{dataset}_{method}*.pickle")
    fX_set = []
    for ii in range(len(data_files_dict[method])):
      ff = data_files_dict[method][ii]
      d = pickle.load(open(ff, "rb"))
    
      if method == "TURBO" or method == "SVGP":
        assert d['model'] == method
      elif method == "DSVGP1":
        assert d['model'] == "DSVGP" and d['num_directions'] == 1
      elif method == "DSVGP2":
        assert d['model'] == "DSVGP" and d['num_directions'] == 2


      fX = d[data_type]
      fXmin = np.minimum.accumulate(fX) if data_type == 'fX' else np.maximum.accumulate(fX)
      fX_set.append(fXmin)
    
    print(f"Averaging {len(fX_set)} trials for {method}\n")

    if deleted_methods == None or method not in deleted_methods:
      label_cur = style_dict[method][0]
      linestyle_cur = style_dict[method][1]
      color_cur = style_dict[method][2]
      # find mean and std of fX_set
      if len(fX_set) > 1:
        fX_mean = np.mean(fX_set, axis=0)
        plt.plot(fX_mean,linewidth=2,markersize=12,
                color=color_cur,label=label_cur,linestyle=linestyle_cur)
        fX_std = np.std(fX_set, axis=0, ddof=0)
        plt.fill_between(range(len(fX_mean)), fX_mean-fX_std, fX_mean+fX_std, color=color_cur, alpha=0.2)
      elif len(fX_set) == 1:
        plt.plot(fX_set[0],linewidth=2,markersize=12,
                 color=color_cur,label=label_cur,linestyle=linestyle_cur)
    
  
  

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
  # plt.title("Convergence of TuRBO for varying Variational GP")
  if data_type == 'fX':
    ylabel = 'Training loss'
    plt.ylim( (10**-2,10**2) )
    if dataset=="squared":
      plt.ylim((0.005,10.))
    plt.yscale("log")
  elif data_type == 'test_acc_list':
    ylabel = 'Test accuracy'
  elif data_type == 'train_acc_list':
    plt.ylim((0.28, 1.05))
    ylabel = 'Training accuracy'
    
  plt.ylabel(ylabel)
  plt.xlabel("Number of evaluations")
  plt.show()
  plt.grid()
  figurename = f"TuRBO_{dataset}_{data_type}.pdf"
  figurepath = os.path.abspath(__file__ + "/../plots/" + figurename)
  plt.savefig(figurepath)
  print("Figure saved:", figurepath)
  plt.close()

if __name__ == "__main__":
  # dataset="squared"
  # methods_list = ["BO", "random"]

  dataset="PubMed"
  methods_list = ["random", "SGD", "TURBO", "SVGP", "DSVGP1", "DSVGP2"]
  
  data_type='fX'
  # data_type='train_acc_list'
  plot_average(dataset, methods_list, data_type, deleted_methods=None)




