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

data = pickle.load( open("./data/rover_plot_data.pickle", "rb" ) )

FONTSIZE=25
MARKERSIZE=20
FIGURESUZE=(10,7)
ALPHA=0.2
LINEWIDTH=5
PADDING=0.1

style_dict = {"TuRBO": ["TuRBO", "dashed", '#ff7f0e'],
              "TuRBO-DPPGPR1": ["TuRBO-DPPGPR1", "solid", '#1f77b4'],
              "TuRBO-DPPGPR2": ["TuRBO-DPPGPR2", "solid", '#d62728'],
              "BO-LCB": ["BO", "dashed", '#8c564b'],
              "Random Search": ["Random", "dotted", "#e377c2"] }

N_method = len(data['labels'])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGURESUZE)
for i in range(N_method):
    method = data['labels'][i]
    mean_data = data['means'][i]
    std_data = data['std'][i]
    ax.plot(mean_data, linewidth=LINEWIDTH,
            color=style_dict[method][2],
            label=style_dict[method][0],
            linestyle=style_dict[method][1])
    ax.fill_between(range(len(mean_data)), 
                    mean_data+std_data, mean_data-std_data, 
                    color=style_dict[method][2], alpha=ALPHA)

plt.ylim([200, 1200])
plt.yticks([300, 500, 700, 900, 1100], fontsize=FONTSIZE)
plt.ylabel('Objective function value', fontsize=FONTSIZE)
plt.xticks([0, 400, 800, 1200, 1600, 2000], fontsize=FONTSIZE)
plt.xlabel("Number of evaluations", fontsize=FONTSIZE)
plt.grid()
plt.legend(fontsize=FONTSIZE-5)

figurename = f"TuRBO_rover.pdf"
figurepath = os.path.abspath(__file__ + "/../plots/" + figurename)
fig.savefig(figurepath, bbox_inches='tight', pad_inches = PADDING)
print("Figure saved:", figurepath)
