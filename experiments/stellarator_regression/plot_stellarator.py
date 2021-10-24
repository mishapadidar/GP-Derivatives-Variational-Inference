
import os
import os.path as osp
import argparse
import pickle
from operator import itemgetter
from argparse import Namespace
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import pylab

ADD_SHARED=True
ncol=1
FONTSIZE=20
MARKERSIZE=15
FIGURESUZE=(10,7)
ALPHA=0.2
ALPHA_MARKER=0.8
LINEWIDTH=4

# ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
#  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

if ADD_SHARED:
    data = pickle.load(open("./data/stellarator_plot_data_p3_with_shared.pickle", "rb" ))
    methods_list = ['PPGPR', 'SVGP', 
                    'DPPGPR2', 'DSVGP2', 'DPPGPR1', 'DSVGP1', 'DPPGPR3', 'DSVGP3',
                    'DPPGPR-Shared1', 'DPPGPR-Shared2', 'DPPGPR-Shared3',
                    'DSVGP-Shared1', 'DSVGP-Shared2', 'DSVGP-Shared3']
    styles = {"PPGPR": ["PPGPR", "solid", '#9467bd', 'o'],
            "SVGP": ["SVGP", "solid", '#1f77b4', '*'],
            "DPPGPR1": ["DPPGPR1", "solid", '#2ca02c', 'v'],
            "DSVGP1": ["DSVGP1","solid", '#8c564b', 'd'],
            "DPPGPR2": ["DPPGPR2", "solid", '#d62728', 'p'],
            "DSVGP2": ["DSVGP2", "solid",  '#ff7f0e', 'X'],
            "DPPGPR3": ["DPPGPR3", "solid", '#e377c2', '^'],
            "DSVGP3": ["DSVGP3", "solid",  '#7f7f7f', '>'],
            "DPPGPR-Shared1": ["DPPGPR-Shared1", "dashed", '#2ca02c', 'v'],
            "DSVGP-Shared1": ["DSVGP-Shared1","dashed", '#8c564b', 'd'],
            "DPPGPR-Shared2": ["DPPGPR-Shared2", "dashed", '#d62728', 'p'],
            "DSVGP-Shared2": ["DSVGP-Shared2", "dashed",  '#ff7f0e', 'X'],
            "DPPGPR-Shared3": ["DPPGPR-Shared3", "dashed", '#e377c2', '^'],
            "DSVGP-Shared3": ["DSVGP-Shared3", "dashed",  '#7f7f7f', '>'],
            }
    ylim = [-2.4, -0.6]
    yticks = [-2.0, -1.6, -1.2, -0.8]
    legend = False
else:
    data = pickle.load(open("./data/stellarator_plot_data_p3.pickle", "rb" ))
    methods_list = ['PPGPR', 'SVGP', 'DPPGPR2', 'DSVGP2', 'DPPGPR1', 'DSVGP1', 'DPPGPR3', 'DSVGP3']
    styles = {"PPGPR": ["PPGPR", "solid", '#9467bd', 'o'],
            "SVGP": ["SVGP", "solid", '#1f77b4', '*'],
            "DPPGPR1": ["DPPGPR1", "solid", '#2ca02c', 'v'],
            "DSVGP1": ["DSVGP1","solid", '#8c564b', 'd'],
            "DPPGPR2": ["DPPGPR2", "solid", '#d62728', 'p'],
            "DSVGP2": ["DSVGP2", "solid",  '#ff7f0e', 'X'],
            "DPPGPR3": ["DPPGPR3", "solid", '#e377c2', '^'],
            "DSVGP3": ["DSVGP3", "solid",  '#7f7f7f', '>'],
            }
    ylim = [-2.4, -1.3]
    yticks = [-2.2, -2.0, -1.8, -1.6, -1.4]
    legend = True



del data["ni"]
del data["nd"]
del data['train_time']
del data['test_time']
del data['mode']
data['rmse'] = np.sqrt(data['mse'])


M_set = [200, 500, 800, 1000, 1200, 1400] # SVGP, DPPGPR, DSVGP1, DPPGPR1,  DSVGP3, DPPGPR3
M_set2 = [198, 498, 798, 999, 1200, 1398] # DSVGP2, DPPGPR2
M_set_dict2 = {198: 200, 498:500, 798:800, 999:1000, 1200:1200, 1398:1400}

data_dict = {}
for method in methods_list:
    # select the partial dataframe  = data[data['run']==method]
    data_dict_cur = {}
    data_dict_full = data[data['run']==method]
    if method.endswith("2"):
        for M in M_set2:
            key = M_set_dict2[M]
            data_dict_cur[key] = data_dict_full[data_dict_full['M']==M]
    else:
        for M in M_set:
            data_dict_cur[M] = data_dict_full[data_dict_full['M']==M]

    data_dict[method] = data_dict_cur


# compelete rmse for mean 
rmse_nll_dict = {}
for method in methods_list:
    rmse_dict = {"mean":[], "std":[]}
    nll_dict = {"mean": [], "std":[]}
    for M in M_set:
        data_cur = data_dict[method][M]
        rmse_mean = np.mean(data_cur['rmse'])
        rmse_std = np.std(data_cur['rmse'])
        nll_mean = np.mean(data_cur['nll'])
        nll_std = np.std(data_cur['nll'])
        rmse_dict["mean"].append(rmse_mean)
        rmse_dict["std"].append(rmse_std)
        nll_dict["mean"].append(nll_mean)
        nll_dict["std"].append(nll_std)
    rmse_nll_dict[method] = {"rmse": rmse_dict, "nll": nll_dict}

 

def plot_stellarator(datatype, ylim=None, yticks=None, logy=False, legend=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGURESUZE)
    for method in methods_list:
        mean_nll = np.array(rmse_nll_dict[method][datatype]['mean'])
        std_nll = np.array(rmse_nll_dict[method][datatype]['std'])
        if ADD_SHARED:
            ax.plot(M_set, mean_nll,
                color=styles[method][2],
                label=styles[method][0],
                linestyle=styles[method][1],
                marker=styles[method][3], 
                markersize=MARKERSIZE,
                alpha=ALPHA_MARKER,
                linewidth=LINEWIDTH,
                )
        else:
            ax.plot(M_set, mean_nll,
                    color=styles[method][2],
                    label=styles[method][0], 
                    marker=styles[method][3], 
                    markersize=MARKERSIZE,
                    alpha=ALPHA_MARKER)
        ax.fill_between(M_set, 
                        mean_nll+std_nll, 
                        mean_nll-std_nll,
                        color=styles[method][2],
                        alpha=ALPHA,
                        )

        ax.set_xlabel("Inducing matrix size",fontsize=FONTSIZE)
        ax.set_xticks(M_set)
        ax.set_xticklabels(M_set,fontsize=FONTSIZE)

        ylabel='NLL' if datatype=='nll' else "RMSE"
        ax.set_ylabel(ylabel, fontsize=FONTSIZE)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks,fontsize=FONTSIZE)
        ax.set_ylim(ylim)
        if legend:
            ax.legend(loc='upper right', fontsize=FONTSIZE-5)

    plt.grid()
    plt.minorticks_off()
    plt.tight_layout()
    if ADD_SHARED:
        figurepath=f"./plots/stellarator_{datatype}_add_shared.pdf"
    else:
        figurepath=f"./plots/stellarator_{datatype}.pdf"
    fig.savefig(figurepath)
    print("Figure saved:", figurepath)

def plot_legend(style_dict, methods_list, ncol):
  plt.clf()
  plt.cla()

  figurename = f"stellarator_legend_add_shared_ncol{ncol}.pdf"
  figurepath = os.path.abspath(__file__ + "/../plots/" + figurename)
  
  color_set = [style_dict[method][2] for method in methods_list]
  linestyle_set = [style_dict[method][1] for method in methods_list]
  label_set = [style_dict[method][0] for method in methods_list]
  marker_set = [style_dict[method][3] for method in methods_list]

  fig, ax = plt.subplots(figsize=(10,8))
  f = lambda ls,c,label,marker: ax.plot([],[], linestyle=ls, color=c, 
                                        label=label, marker=marker, 
                                        markersize=MARKERSIZE*0.68,
                                        # linewidth=LINEWIDTH,
                                        )[0]
  handles = [f(linestyle_set[i], color_set[i], label_set[i], marker_set[i]) for i in range(len(methods_list))]
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


# plot_stellarator('nll', ylim=ylim, yticks=yticks, 
#          logy=False, legend=legend)
if ADD_SHARED:
    plot_legend(styles, methods_list, ncol=ncol)

