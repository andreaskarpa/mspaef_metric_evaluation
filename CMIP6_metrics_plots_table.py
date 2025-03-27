#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:15:42 2025

@author: andreaskarpasitis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cmocean as cm
import random
import matplotlib as mpl

mpl.rcParams["font.family"] = "Arial" #"Ani"

def generate_random_numbers(n, seed=None):
  """Generates a list of n random numbers between 0 and 1.

  Args:
    n: The number of random numbers to generate.
    seed: An optional seed value for reproducibility.

  Returns:
    A list of n random numbers between 0 and 1.
  """
  if seed is not None:
    random.seed(seed)
  return [random.random() for _ in range(n)]

path_in = '/path/to/somewhere/'   ### path where the csv file are
path = '/path/to/somewhere/'     ### path to output the figure and the ranked csv files
file_name_pr = 'Metrics_CMIP6_models_pr.csv'  ### name of csv files containing the precipitation metric data
file_name_tas = 'Metrics_CMIP6_models_tas.csv'   ### name of csv files containing the temperature metric data

csv_pr = pd.read_csv(os.path.join(path_in,file_name_pr))
csv_tas = pd.read_csv(os.path.join(path_in,file_name_tas))

mspaef_pr = csv_pr['MSPAEF'].values
mspaef_tas = csv_tas['MSPAEF'].values

fig, axes = plt.subplots(1,2,figsize=(19,12),width_ratios=[3, 1])

colors = generate_random_numbers(33, seed=1)
n = np.arange(1,34,1)

ax=axes[0]
texts = []
ax.scatter(mspaef_pr,mspaef_tas,c='grey',cmap=cm.cm.deep,s=50,edgecolors='black',alpha=0.8)

for i, txt in enumerate(n):
    texts.append(ax.annotate(str(txt), (mspaef_pr[i]+0.0015, mspaef_tas[i]+0.0015),fontsize=25))

ax.set_xlim(0.74,0.88)
ax.set_ylim(0.81,1)
ax.set_yticks([0.85,0.9,0.95,1],[0.85,0.9,0.95,1])
ax.set_xticks([0.75,0.8,0.85],[0.75,0.8,0.85])
ax.set_xlabel('Precipitation MSPAEF',fontsize=30)
ax.set_ylabel('Temperature MSPAEF',fontsize=30)

ax.grid()
ax.tick_params(axis='both', which='major', labelsize=25)


ax = axes[1]
models_legend_text = '  1 : ACCESS-CM2\n  2 : ACCESS-ESM1-5\n  3 : AWI-CM-1-1-MR\n  4 : BCC-CSM2-MR\n  5 : CESM2-WACCM\n  6 : CESM2\n  7 : CIESM\n  8 : CMCC-CM2-SR5\n  9 : CNRM-CM6-1-HR\n 10 : CNRM-CM6-1\n 11 : CNRM-ESM2-1\n 12 : CanESM5-CanOE\n 13 : CanESM5\n 14 : EC-Earth3-Veg\n 15 : EC-Earth3\n 16 : FGOALS-f3-L\n 17 : FGOALS-g3\n 18 : GFDL-ESM4\n 19 : GISS-E2-1-G\n 20 : HadGEM-GC31-LL\n 21 : INM-CM4-8\n 22 : INM-CM5-0\n 23 : IPSL-CM6A-LR\n 24 : KACE-1-0-G\n 25 : MIROC-ES2L\n 26 : MIROC6\n 27 : MPI-ESM1-2-HR\n 28 : MPI-EM1-2-LR\n 29 : MRI-ESM2-0\n 30 : NESM3\n 31 : NorESM2-LM\n 32 : NorESM2-MM\n 33 : UKESM1-0-LL'
ax.text(0.00, 1, models_legend_text, transform=ax.transAxes, fontsize=22,
        verticalalignment='top')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(path,'Tas_vs_pr_CMIP6.pdf'),dpi=300)


csv_pr['rank_mspaef'] = csv_pr['MSPAEF'].rank(ascending=False, method='min').astype(int)
csv_pr['rank_spaef'] = csv_pr['SPAEF'].rank(ascending=False, method='min').astype(int)
csv_pr['rank_wspaef'] = csv_pr['WSPAEF'].rank(ascending=True, method='min').astype(int)
csv_pr['rank_esp'] = csv_pr['Esp'].rank(ascending=False, method='min').astype(int)
csv_pr['Model'] = csv_pr['Model'] + 1

csv_tas['rank_mspaef'] = csv_tas['MSPAEF'].rank(ascending=False, method='min').astype(int)
csv_tas['rank_spaef'] = csv_tas['SPAEF'].rank(ascending=False, method='min').astype(int)
csv_tas['rank_wspaef'] = csv_tas['WSPAEF'].rank(ascending=True, method='min').astype(int)
csv_tas['rank_esp'] = csv_tas['Esp'].rank(ascending=False, method='min').astype(int)
csv_tas['Model'] = csv_tas['Model'] + 1

csv_pr.to_csv(os.path.join(path_in,'Metrics_CMIP6_models_pr_rank.csv'))
csv_tas.to_csv(os.path.join(path_in,'Metrics_CMIP6_models_tas_rank.csv'))