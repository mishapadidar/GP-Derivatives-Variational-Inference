
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


# read sin5_plot_data.pickle
data = pickle.load( open("./data/stellarator_plot_data.pickle", "rb" ) )
del data["ni"]
del data["nd"]
del data['train_time']
del data['test_time']
del data['mode']

data['rmse'] = np.sqrt(data['mse'])


methods_list = ['PPGPR', 'SVGP', 'DPPGPR2', 'DSVGP2', 'DPPGPR1', 'DSVGP1']
M_set = [200, 500, 800, 1000, 1200, 1400]
M_set2 = [198, 498, 798, 999, 1200, 1398]
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
        
        
FONTSIZE=25
MARKERSIZE=20
FIGURESUZE=(10,7)
ALPHA=0.2

styles = {"PPGPR": ["PPGPR", "solid", '#9467bd', 'o'],
          "SVGP": ["SVGP", "solid", '#1f77b4', '*'],
          "DPPGPR2": ["DPPGPR2", "solid", '#d62728', 'p'],
          "DSVGP2": ["DSVGP2", "solid",  '#ff7f0e', 'X'],
          "DPPGPR1": ["DPPGPR1", "solid", '#2ca02c', 'v'],
          "DSVGP1": ["DSVGP1","solid", '#8c564b', 'd']}

def plot_stellarator(datatype, ylim=None, yticks=None, logy=False, legend=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGURESUZE)
    for method in methods_list:
        mean_nll = np.array(rmse_nll_dict[method][datatype]['mean'])
        std_nll = np.array(rmse_nll_dict[method][datatype]['std'])
        ax.plot(M_set, mean_nll,
                color=styles[method][2],
                label=styles[method][0], 
                marker=styles[method][3], 
                markersize=MARKERSIZE)
        ax.fill_between(M_set, 
                        mean_nll+std_nll, 
                        mean_nll-std_nll,
                        color=styles[method][2],
                        alpha=ALPHA)

        ax.set_xlabel("Inducing matrix size",fontsize=FONTSIZE)
#         ax.set_xscale('log')
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
    figurepath=f"./plots/stellarator_{datatype}.pdf"
    fig.savefig(figurepath)
    print("Figure saved:", figurepath)


plot_stellarator('nll', ylim=[-2.4, -1.3], yticks=[-2.2, -2.0, -1.8, -1.6, -1.4], 
         logy=False, legend=True)

# plot_stellarator('rmse', ylim=[2e-2, 5e-2], 
#                  yticks=[3e-2, 4e-2], 
#                  logy=False, legend=True)