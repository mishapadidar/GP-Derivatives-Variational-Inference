from sys import meta_path
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
# import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import glob
import os

ADD_SHARED=True
FONTSIZE=25
MARKERSIZE=20
FIGURESUZE=(10,7)
ALPHA=0.2
LINEWIDTH=5
PADDING=0.1

if ADD_SHARED:
  style_dict = {"SGD": ["GD", "dotted", '#2ca02c'],
                  "TURBO": ["TuRBO", "dashed", '#ff7f0e'],
                  "DSVGP1": ["TuRBO-DPPGPR1", "solid", '#1f77b4'],
                  "DSVGP2": ["TuRBO-DPPGPR2", "solid", '#d62728'],
                  "DSVGP3": ["TuRBO-DPPGPR3", "solid", '#e377c2'],
                  "DSVGP_shared1": ["TuRBO-DPPGPR-Shared1", "dotted", '#1f77b4'],
                  "DSVGP_shared2": ["TuRBO-DPPGPR-Shared2", "dotted", '#d62728'],
                  "DSVGP_shared3": ["TuRBO-DPPGPR-Shared3", "dotted", '#e377c2'],
                  "SVGP": ["TuRBO-PPGPR", "dashed", '#9467bd'],
                  "BO": ["BO", "dashed", '#8c564b'],
                  "random": ["Random", "dotted", "#7f7f7f"] }
else:
  style_dict = {"SGD": ["GD", "dotted", '#2ca02c'],
                  "TURBO": ["TuRBO", "dashed", '#ff7f0e'],
                  "DSVGP1": ["TuRBO-DPPGPR1", "solid", '#1f77b4'],
                  "DSVGP2": ["TuRBO-DPPGPR2", "solid", '#d62728'],
                  "DSVGP3": ["TuRBO-DPPGPR3", "solid", '#e377c2'],
                  "SVGP": ["TuRBO-PPGPR", "dashed", '#9467bd'],
                  "BO": ["BO", "dashed", '#8c564b'],
                  "random": ["Random", "dotted", "#7f7f7f"] }


def plot_average(style_dict, dataset, methods_list, data_type='fX', deleted_methods=None):
  assert data_type == 'fX' or data_type == 'train_acc_list' or data_type == 'test_acc_list'

  # sort to fix color for each method
  colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  
  # collect data for each method
  data_files_dict = {}
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGURESUZE)

  for i, method in enumerate(methods_list):
    data_files_dict[method] = glob.glob(f"./results/{dataset}_{method}*.pickle")
    fX_set = []
    for ii in range(len(data_files_dict[method])):
      ff = data_files_dict[method][ii]
      d = pickle.load(open(ff, "rb"))
      if method == "TURBO" or method == "SVGP":
        assert d['model'] == method
      elif method == "DSVGP1" or method == "DSVGP_shared1":
        assert d['model'].startswith("DSVGP") and d['num_directions'] == 1
      elif method == "DSVGP2" or method == "DSVGP_shared2":
        assert d['model'].startswith("DSVGP") and d['num_directions'] == 2
      elif method == "DSVGP3" or method == "DSVGP_shared3":
        assert d['model'].startswith("DSVGP") and d['num_directions'] == 3

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
        ax.plot(fX_mean,linewidth=LINEWIDTH,
                color=color_cur,label=label_cur,linestyle=linestyle_cur)
        fX_std = np.std(fX_set, axis=0, ddof=0)
        ax.fill_between(range(len(fX_mean)), fX_mean-fX_std, fX_mean+fX_std, color=color_cur, alpha=ALPHA)
      elif len(fX_set) == 1:
        ax.plot(fX_set[0],linewidth=LINEWIDTH,
                 color=color_cur,label=label_cur,linestyle=linestyle_cur)
    
  
  

  # plot
  rc = {'figure.figsize':(12,6),
        'axes.facecolor':'white',
        'axes.grid' : True,
        'grid.color': '.8',
        'font.family':'Times New Roman',
        'font.size' : FONTSIZE}
  plt.rcParams.update(rc)
  if data_type == 'fX':
    ylabel = 'Training loss'
    plt.ylim( (0.65e-2,1.35e2) )
    plt.yscale("log")
  elif data_type == 'test_acc_list':
    ylabel = 'Test accuracy'
  elif data_type == 'train_acc_list':
    plt.ylim((0.28, 1.05))
    ylabel = 'Training accuracy'
  
  plt.xticks(fontsize=FONTSIZE)
  plt.yticks(fontsize=FONTSIZE)
  plt.ylabel(ylabel, fontsize=FONTSIZE)
  plt.xlabel("Number of evaluations", fontsize=FONTSIZE)
  # box = plt.get_position()
  # plt.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  # plt.legend(loc='best',prop={'size': fontsize})
  
  plt.grid()
  # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

  if ADD_SHARED:
    figurename = f"TuRBO_{dataset}_{data_type}_add_shared.pdf"
  else:
    figurename = f"TuRBO_{dataset}_{data_type}.pdf"
  figurepath = os.path.abspath(__file__ + "/../plots/" + figurename)
  fig.savefig(figurepath, bbox_inches = 'tight', pad_inches = PADDING)
  print("Figure saved:", figurepath)

def plot_legend(style_dict, dataset, methods_list):
  plt.clf()
  plt.cla()

  if ADD_SHARED:
    figurename = f"TuRBO_{dataset}_legend_add_shared.pdf"
  else:
    figurename = f"TuRBO_{dataset}_legend.pdf"
  figurepath = os.path.abspath(__file__ + "/../plots/" + figurename)
  
  color_set = [style_dict[method][2] for method in methods_list]
  linestyle_set = [style_dict[method][1] for method in methods_list]
  label_set = [style_dict[method][0] for method in methods_list]
  ncol = len(style_dict)//2

  fig, ax = plt.subplots(figsize=(10,8))
  f = lambda ls,c,label: ax.plot([],[], linewidth=LINEWIDTH+1, linestyle=ls, color=c, label=label)[0]
  handles = [f(linestyle_set[i], color_set[i], label_set[i]) for i in range(len(methods_list))]
  #legend
  LABEL_SIZE=15
  figsize = (5, 1)
  fig_leg = plt.figure(figsize=figsize)
  legend_properties = {'weight': 'bold', 'size': LABEL_SIZE}
  ax_leg = fig_leg.add_subplot(111)

  ax_leg.set_facecolor('white')
  ax_leg.grid(False)
  ax_leg.set_axis_off()
  ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=ncol, prop=legend_properties, facecolor="white", edgecolor="grey")
  fig_leg.savefig(figurepath, bbox_inches = 'tight')

  print("Figure saved:", figurepath)

if __name__ == "__main__":

  # dataset="squared"
  # methods_list = ["BO", "random"]

  dataset="PubMed"
  if ADD_SHARED:
    methods_list = ["random", "SGD", "BO", "TURBO", "SVGP", 
                    "DSVGP1", "DSVGP2", "DSVGP3",
                    "DSVGP_shared1", "DSVGP_shared2", "DSVGP_shared3"]
  else:
    methods_list = ["random", "SGD", "BO", "TURBO", "SVGP", 
                    "DSVGP1", "DSVGP2", "DSVGP3"]
  
  data_type='fX'
  # data_type='train_acc_list'
  plot_average(style_dict, dataset, methods_list, data_type, deleted_methods=None)
  plot_legend(style_dict, dataset, methods_list)



