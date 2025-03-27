#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:12:18 2025

@author: andreaskarpasitis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib as mpl
import copy


mpl.rcParams["font.family"] = "Arial" #"Ani"

path_in = '/path/to/somewhere/'    ### path where the input ranked csv files are
path = '/path/to/somewhere/'    ### path to output figure
file_name_pr = 'Metrics_CMIP6_models_pr_rank.csv'  ### name of ranked pr csv file for precipitation
file_name_tas = 'Metrics_CMIP6_models_tas_rank.csv'   ### name of ranked pr csv file for temperature

csv_pr = pd.read_csv(os.path.join(path_in,file_name_pr))
csv_tas = pd.read_csv(os.path.join(path_in,file_name_tas))

mspaef_pr = csv_pr['MSPAEF'].values
mspaef_tas = csv_tas['MSPAEF'].values



csv_pr_rank_mspaef = csv_pr['rank_mspaef']
csv_pr_rank_spaef = csv_pr['rank_spaef']
csv_pr_rank_wspaef = csv_pr['rank_wspaef']
csv_pr_rank_esp = csv_pr['rank_esp']

csv_tas_rank_mspaef = csv_tas['rank_mspaef']
csv_tas_rank_spaef = csv_tas['rank_spaef']
csv_tas_rank_wspaef = csv_tas['rank_wspaef']
csv_tas_rank_esp = csv_tas['rank_esp']
le = len(csv_tas_rank_esp)
pr_mspaef_wspaef = np.sum(np.abs(csv_pr_rank_mspaef - csv_pr_rank_wspaef))/le
pr_spaef_wspaef = np.sum(np.abs(csv_pr_rank_spaef - csv_pr_rank_wspaef))/le
pr_esp_wspaef = np.sum(np.abs(csv_pr_rank_esp - csv_pr_rank_wspaef))/le
pr_mspaef_spaef = np.sum(np.abs(csv_pr_rank_mspaef - csv_pr_rank_spaef))/le
pr_esp_spaef = np.sum(np.abs(csv_pr_rank_esp - csv_pr_rank_spaef))/le
pr_mspaef_esp = np.sum(np.abs(csv_pr_rank_mspaef - csv_pr_rank_esp))/le

tas_mspaef_wspaef = np.sum(np.abs(csv_tas_rank_mspaef - csv_tas_rank_wspaef))/le
tas_spaef_wspaef = np.sum(np.abs(csv_tas_rank_spaef - csv_tas_rank_wspaef))/le
tas_esp_wspaef = np.sum(np.abs(csv_tas_rank_esp - csv_tas_rank_wspaef))/le
tas_mspaef_spaef = np.sum(np.abs(csv_tas_rank_mspaef - csv_tas_rank_spaef))/le
tas_esp_spaef = np.sum(np.abs(csv_tas_rank_esp - csv_tas_rank_spaef))/le
tas_mspaef_esp = np.sum(np.abs(csv_tas_rank_mspaef - csv_tas_rank_esp))/le



column = ['MSPAEF','SPAEF','WSPAEF','Esp']
rows = ['MSPAEF','SPAEF','WSPAEF','Esp']
row1 = [0,pr_mspaef_spaef,pr_mspaef_wspaef,pr_mspaef_esp]
row2 = [tas_mspaef_spaef,0,pr_spaef_wspaef,pr_esp_spaef]
row3 = [tas_mspaef_wspaef,tas_spaef_wspaef,0,pr_esp_wspaef]
row4 = [tas_mspaef_esp,tas_esp_spaef,tas_esp_wspaef,0]
data =[row1,row2,row3,row4]
dataframe = pd.DataFrame(data, column,rows)
dataframe = dataframe.round(2)
dataframe_copy = copy.deepcopy(dataframe)
for metric in column:
    dataframe_copy.loc[metric,metric] = '-'
    
# Create masks for upper and lower triangles
mask_upper = np.triu(np.ones(dataframe.shape, dtype=bool), k=1)  # Upper triangle (excluding diagonal)
mask_lower = np.tril(np.ones(dataframe.shape, dtype=bool), k=-1) # Lower triangle (excluding diagonal)

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 6))

# Define color maps
cmap_upper = "Blues"  # Change as needed
cmap_lower = "Reds"

# Plot upper triangle
upper_values = np.where(mask_upper, dataframe, np.nan)
mesh_upper = ax.pcolormesh(upper_values, cmap=cmap_upper, edgecolors='k', shading='flat',vmin=0,vmax=10)

# Plot lower triangle
lower_values = np.where(mask_lower, dataframe, np.nan)
mesh_lower = ax.pcolormesh(lower_values, cmap=cmap_lower, edgecolors='k', shading='flat',vmin=0,vmax=10)

# Add colorbars
cbar_ax_upper = fig.add_axes([0.95, 0.55, 0.03, 0.35])
cbar_upper = fig.colorbar(mesh_upper, cax=cbar_ax_upper)
cbar_upper.set_label("Precipitation",fontsize=15)
cbar_upper.ax.tick_params(labelsize=15) 

cbar_ax_lower = fig.add_axes([0.95, 0.12, 0.03, 0.35])  # Positioning for lower colorbar
cbar_lower = fig.colorbar(mesh_lower, cax=cbar_ax_lower)
cbar_lower.set_label("Temperature", fontsize=15)
cbar_lower.ax.tick_params(labelsize=15) 

# Add annotations
for i in range(len(column[:])):
    for j in range(len(column[:])):
        if i != j:
            value = dataframe.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, f"{value:.1f}", ha='center', va='center', color='black',fontsize=15)

# Format plot
ax.set_xticks(np.arange(len(column)) + 0.5)
ax.set_yticks(np.arange(len(column)) + 0.5)
ax.set_xticklabels(column, rotation=0, ha='center',fontsize=15)
ax.set_yticklabels(column,fontsize=15)
ax.set_xlim(0, len(column))
ax.set_ylim(len(column), 0)  # Reverse y-axis for correct alignment
ax.set_title("Rank Differences Between Pairs of Metrics", fontsize=15)
plt.savefig(os.path.join(path,'Full_matrix_ranking_differences.pdf'),dpi=300, bbox_inches='tight')
plt.close()