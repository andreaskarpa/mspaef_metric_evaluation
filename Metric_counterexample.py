#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:06:03 2024

@author: andreaskarpasitis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.special import kv, gamma
from scipy import stats
from sklearn import metrics
import os
import cmocean as cm 
import warnings
import matplotlib as mpl
warnings.filterwarnings("ignore")

mpl.rcParams["font.family"] = "Arial" #"Ani"
# Define Matern covariance function
def matern_covariance(r, nu=1.5, length_scale=1.0):
    """
    Generate Covariance matrix using the Mattern Covariance function
    
    Parameters:
    - r: np.array, distance matrix
    - nu: smooth paramater
    - length_scale: length scale
    
    Returns:
    - matern: np.array, covariance matrix 
    """
    factor = (2 ** (1 - nu)) / gamma(nu)
    scaled_r = np.sqrt(2 * nu*r)
    second_term = (scaled_r **nu) / length_scale
    matern = factor * (second_term) * kv(nu, scaled_r)
    matern[scaled_r == 0] = 1  # Fix the value at zero
    return matern

# Create distance matrix for 5x5 grid
def create_distance_matrix(grid_size=5):
    """
    Generate distance matrix for a specifiec grid size
    
    Parameters:
    - grid_size: np.int, size of the grid for the synthetic field
    
    Returns:
    - dist_matrix: np.array, distance matrix 
          encoding the distance between each combination of grid-points.
    """
    coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])
    dist_matrix = squareform(pdist(coords))
    return dist_matrix

# Generate synthetic field using Matern covariance
def generate_synthetic_field(dist_matrix, grid_size=5, std_dev=1.0, mean=0.0, length_scale=1.0, seed=None):
    """
    Generate synthetic field x with normal distribution
    with specified mean and std
    
    Parameters:
    - dist_matrix: np.array, the distance matrix.
    - grid_size: float, the size of each of the dimensions of the matrix.
    - std: float, desired standard deviation ratio of x
    - mean: float, desired mean value of x.
    - seed: int or None, to get the same results using random data each time the function is called, use a seed,
                         otherwise use None to get different results each time the function is called
    
    Returns:
    - field: np.array, synthetic field  
          with specified std and mean value.
    """
    cov_matrix = matern_covariance(dist_matrix, length_scale=length_scale)
    if seed != None:
        np.random.seed(int(seed))
    d = dist_matrix.shape[0]
    sigma_current = np.sqrt(np.trace(cov_matrix) / d)
    field = np.random.multivariate_normal(mean * np.ones(dist_matrix.shape[0]), (std_dev/sigma_current) ** 2 * cov_matrix) + 0
    return field.reshape((grid_size, grid_size))  # Reshape back to 5x5 grid

# Generate correlated field with specified correlation, standard deviation ratio, and bias
def generate_correlated_field(x, target_corr=0.5, std_ratio=1.0, bias=0.0):
    """
    Generate a correlated field y with normal distribution
    with control over standard deviation ratio and bias.
    
    Parameters:
    - x: np.array, the initial synthetic field.
    - target_corr: float, desired Pearson correlation between x and y.
    - std_ratio: float, desired standard deviation ratio between x and y (σ_y / σ_x).
    - bias: float, desired bias to add to y.
    
    Returns:
    - y: np.array, correlated synthetic field  
          with specified std ratio, correlation and bias.
    """
    x_flat = x.flatten()
    y_flat = std_ratio * (target_corr * (x_flat - np.mean(x_flat)) + np.sqrt(1 - target_corr ** 2) * np.random.randn(*x_flat.shape)*np.std(x_flat)) + np.mean(x_flat)
    y_flat += bias  # Apply the bias
    return y_flat.reshape(x.shape)

# Calculate SPAEF metric
def calculate_spaef(x, y):
    '''
    Calculated the four metrics SPAEF, WSPAEF, MSPAEF and Esp
    Parameters
    - x: 1D np.array, the flattened array of compariosn dataset.
    - y: 1D np.array, the flattened array of model dataset.
    
    Returns:
    - spaef : SPAEF metric value.
    - wspaef : WSPAEF metric value
    - mspaef : MSPAEF metric value
    - esp : Esp metric value
    '''

    data_min = min(np.min(x),np.min(y))
    data_max = max(np.max(x),np.max(y))
    
    n_bins=100
    # Calculate the bin edges, ensuring they are integers
    bin_edges = np.linspace(
        int(np.floor(data_min)), int(np.ceil(data_max)), n_bins + 1)  ### Bin edges for histogram using original data
    data_comp_pdf, _ = np.histogram(x.flatten(), bins=bin_edges, density=False)  ### Histogram bins for original data of comparison dataset
    data_pdf, _ = np.histogram(y.flatten(), bins=bin_edges, density=False)  ### Histogram bins for original data of model dataset
    
    
    #######################################
    # Histogram, removing the mean
    x_removed_mean = x - np.mean(x)
    y_removed_mean = y - np.mean(y)
    
    data_min_mean = min(np.min(x_removed_mean),np.min(y_removed_mean))
    data_max_mean = max(np.max(x_removed_mean),np.max(y_removed_mean))
    n_bins=100
    # Calculate the bin edges, ensuring they are integers
    bin_edges_mean = np.linspace(
        int(np.floor(data_min_mean)), int(np.ceil(data_max_mean)), n_bins + 1)  ### Bin edges for histogram using removed mean of data
    data_comp_mean_pdf, _ = np.histogram(x_removed_mean.flatten(), bins=bin_edges_mean, density=False)  ### Histogram bins for removed mean of data of comparison dataset
    data_mean_pdf, _ = np.histogram(y_removed_mean.flatten(), bins=bin_edges_mean, density=False)  ### Histogram bins for removed mean of data of model dataset
    ########################################## Histogram bins for removed mean of data of comparison dataset
    
    
    
    ########################################
    zscore_data_comp = stats.zscore(x.flatten())    ### Z-score of comparison dataset
    zscore_data = stats.zscore(y.flatten())   ### Z-score of model dataset
    ## Histogram of the zscores
    # Define the bin size factor
    bin_factor = 0.1
    
    # Define the range of precipitation values (log scale)
    min_zscore = -4
    max_zscore = 4
    
    # Calculate the bin edges
    bins = []
    value = min_zscore
    while value <= max_zscore:
        bins.append(value)
        value += bin_factor
    bins = np.array(bins)  ### Bin edges for histogram using z-score of data
    
    data_comp_zscore_pdf, _ = np.histogram(zscore_data_comp, bins=bins, density=False)  ### Histogram bins for z-score of data of comparison dataset
    data_zscore_pdf, _ = np.histogram(zscore_data, bins=bins, density=False)  ### Histogram bins for z-score of data of model dataset
    #######################################
    
    
    gamma_data_zscore = np.sum(np.minimum(data_zscore_pdf,data_comp_zscore_pdf))/np.sum(data_comp_zscore_pdf)  ### Overlap of z-score histograms. For use in SPAEF
    mean_ratio = np.mean(x) / np.mean(y)   ### Ratio of mean values of comparison vs model datasets
    std_ratio = np.std(x) / np.std(y)   ### Ratio of standard deviation values of comparison vs model datasets
    b = mean_ratio/std_ratio   #### For use in SPAEF and Esp metrics
    
    q75, q25 = np.percentile(x, [75 ,25])
    iqr_x = q75 - q25
    
    corr = np.corrcoef(x.flatten(), y.flatten())[0, 1] #### Pearson Correlation coefficient, for use in SPAEF, MSPAEF and WSPAEF metrics
    rs = stats.spearmanr(x.flatten(),y.flatten())[0] #### Spearman Correlation coefficient, for use in Esp metric
    
    nmse = np.sqrt(np.sum((x-y)**2)/(len(x.flatten())))/iqr_x #### Normalized (over std of comparison dataset) Root Mean Square Error of original data  
    wd_data = stats.wasserstein_distance(bin_edges[:-1],bin_edges[:-1], data_comp_pdf, data_pdf )  ##### Wasserstein distance of histograms of the original data
    alpha = 1 - metrics.root_mean_squared_error(zscore_data_comp,zscore_data)   ### Root Mean Square Error of z-score data, for use in Esp
    
    
    spaef = np.sqrt((1 - b) ** 2 + (1 - gamma_data_zscore) ** 2 + (1 - corr) ** 2)   ### SPAEF metric, modified so that 0 indicates perfect model preformance 
    wspaef = np.sqrt( (1 - corr) ** 2 + (1 - 1/std_ratio) ** 2 + wd_data**2)         ### WSPAEF metric
    mspaef = np.sqrt((nmse)**2 + (np.sqrt((std_ratio**2 - 1)**2 + (1/std_ratio**2 - 1)**2))**2 + ((np.abs(np.mean(x)-np.mean(y))/iqr_x))**2 + (1-corr)**2)/np.sqrt(4)   ### MSPAEF metric, modified so that 0 indicates perfect model preformance    
    esp = np.sqrt((rs-1)**2 + (b-1)**2 + (alpha - 1)**2)                             ### Esp metric, modified so that 0 indicates perfect model preformance 
    
    return spaef,wspaef, mspaef, esp


def generate_skewed_correlated_field(x, rho,
                                     lambda_ratio=1.0, delta_bias=0.0):
    """
    Generate a correlated field y with log-normal distribution, preserving skewness similar to x,
    with control over standard deviation ratio and bias.
    
    Parameters:
    - x: np.array, the initial synthetic field (already transformed to log-normal).
    - rho: float, desired Pearson correlation between x and y (in normal space).
    - mean_x: float, mean of the underlying normal distribution for x.
    - sigma_x: float, standard deviation of the underlying normal distribution for x.
    - lambda_ratio: float, desired standard deviation ratio (σ_y / σ_x).
    - delta_bias: float, desired bias to add to y (in log-normal space).
    
    Returns:
    - y: np.array, correlated synthetic field with similar skewness to x, 
          with specified std ratio and bias.
    """
    # Step 1: Transform the log-normal field x back to its underlying normal distribution
    x_normal = np.log(x)
    mean_x = np.mean(x_normal)
    # Step 2: Generate a new independent standard normal field v
    v = np.random.normal(0, 1, x.shape)
    
    # Step 3: Create the correlated normal field y_normal with correlation rho
    y_normal = rho * (x_normal - mean_x) + np.sqrt(1 - rho ** 2) * v * np.std(x_normal)
    
    # Step 4: Adjust the standard deviation of y_normal based on the desired lambda_ratio
    y_normal = y_normal* lambda_ratio + mean_x
    
    # Step 5: Apply the log-normal transformation to obtain the skewed field y
    y = np.exp(y_normal)

    # Step 6: Add the desired bias to y in log-normal space
    y += delta_bias
    
    y = np.where(y>0,y,0)
    
    return y
# %%

out_path = '/path/to/somewhere/' ## Path to output the figures

# Parameters
num_realizations = 1
grid_size = 30  ### Size of grids to use for each example
correlations = [-0.9, -0.3, 0.3, 0.9]  # Correlation values
std_ratios = [0.1, 0.7, 1.0, 1.3, 1.9]  # Variation ratios
biases = [0.0, 1.0, 5.0]  # Bias values

# Distance matrix for Matern covariance
dist_matrix = create_distance_matrix(grid_size)

# Containers to store results
spaef_results = {"Good": [], "Bad": []}


x = generate_synthetic_field(dist_matrix, grid_size=grid_size, std_dev=70.0, mean=200.0, length_scale=1.0, seed=0) 

x = np.where(x>0,x,0)

y_good = generate_correlated_field(x,target_corr=0.85,std_ratio=1.1,bias=-10)
y_good = np.where(y_good>0,y_good,0)


y_bad = generate_correlated_field(x,target_corr=-0.4,std_ratio=1.0,bias=0)

y_bad = np.where(y_bad>0,y_bad,0)
# %%


fig = plt.figure()
ax = plt.axes()
c = ax.pcolormesh(x,vmin=0,vmax=300,cmap=cm.cm.deep)
plt.colorbar(c,ax=ax,orientation='vertical', location='right')
plt.show()

fig = plt.figure()
ax = plt.axes()
c = ax.pcolormesh(y_good,vmin=0,vmax=300,cmap=cm.cm.deep)
plt.colorbar(c,ax=ax,orientation='vertical', location='right')
plt.show()

fig = plt.figure()
ax = plt.axes()
c = ax.pcolormesh(y_bad,vmin=0,vmax=300,cmap=cm.cm.deep)
plt.colorbar(c,ax=ax,orientation='vertical', location='right')
plt.show()
# %%


fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharex=True, sharey=True, gridspec_kw={'width_ratios': [1, 1, 1, 0.01]})
fig.suptitle("Example 1", fontsize=16)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

c = axes[0].pcolormesh(x,vmin=0,vmax=300,cmap=cm.cm.deep)
#plt.colorbar(c,ax=axes[0],orientation='vertical', location='right')
axes[0].set_title('Observations')

c = axes[1].pcolormesh(y_good,vmin=0,vmax=300,cmap=cm.cm.deep)
#plt.colorbar(c,ax=axes[1],orientation='vertical', location='right')
axes[1].set_title('Model A')

c = axes[2].pcolormesh(y_bad,vmin=0,vmax=300,cmap=cm.cm.deep)
axes[2].set_title('Model B')

plt.colorbar(c,ax=axes[3],orientation='vertical', location='right',fraction=4, pad=0.04)
axes[3].axis('off')
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(out_path,'Example1.pdf'), dpi=300)
plt.close()
# %%



for y in [y_good,y_bad]:   
    spaef_val, wspaef_val, mspaef_val, esp_val = calculate_spaef(x,y)
    print('WSPAEF: {}, SPAEF: {}, MSPAEF: {}, E_sp: {}'.format(wspaef_val,spaef_val,mspaef_val,esp_val))
# %%
# Generate SPAEF values for different parameter combinations
for i in range(20):
    x = generate_synthetic_field(dist_matrix, grid_size=grid_size, std_dev=70.0, mean=200.0, length_scale=1.0) 

    x = np.where(x>0,x,0)
    y_good = generate_correlated_field(x,target_corr=0.85,std_ratio=1.1,bias=-10)
    y_good = np.where(y_good>0,y_good,0)
    y_bad = generate_correlated_field(x,target_corr=-0.4,std_ratio=1.0,bias=0)
    y_bad = np.where(y_bad>0,y_bad,0)
    
    spaef_val, wspaef_val, mspaef_val, esp_val = calculate_spaef(x,y_good)
    spaef_results['Good'].append((spaef_val,wspaef_val,mspaef_val,esp_val))
    spaef_val, wspaef_val, mspaef_val, esp_val = calculate_spaef(x,y_bad)
    spaef_results['Bad'].append((spaef_val,wspaef_val,mspaef_val,esp_val))
    
fig, axes = plt.subplots(1,4,figsize=(12,6))
for j, metric in enumerate(['WSPAEF', 'MSPAEF', 'SPAEF', '$E_{sp}$']):
    if metric == 'WSPAEF':
        good_results = [f[1] for f in spaef_results['Good']]
        bad_results = [f[1] for f in spaef_results['Bad']]
    elif metric == 'MSPAEF':
        good_results = [f[2] for f in spaef_results['Good']]
        bad_results = [f[2] for f in spaef_results['Bad']]
    elif metric == 'SPAEF':
        good_results = [f[0] for f in spaef_results['Good']]
        bad_results = [f[0] for f in spaef_results['Bad']]
    else:
        good_results = [f[3] for f in spaef_results['Good']]
        bad_results = [f[3] for f in spaef_results['Bad']]
    ax = axes[j]
    ax.boxplot(good_results, vert=True, positions=[1])
    ax.boxplot(bad_results, vert=True, positions=[2])
    ax.set_title(metric)
    ax.set_xticks([1,2],['Model A','Model B'], fontsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(out_path,'Example1_metrics.pdf'), dpi=300)
plt.close()



#####
######
### Disprove the other 2 #####
######
#####

# Containers to store results
spaef_results = {"Good": [], "Bad": []}


x = generate_synthetic_field(dist_matrix, grid_size=grid_size, std_dev=1.0, mean=6.0, length_scale=1.0, seed=0) 

x = np.where(x>0,x,0)
disturbed_field = generate_synthetic_field(dist_matrix, grid_size=grid_size, std_dev=1, mean=0.0, length_scale=1.0) 
y_good = generate_correlated_field(x,target_corr=0.6,std_ratio=1.1,bias=0.0)
y_good = np.where(y_good>0,y_good,0)


y_bad = generate_correlated_field(x,target_corr=0.8,std_ratio=2.0,bias=7.5)

y_bad = np.where(y_bad>0,y_bad,0)
# %%


fig = plt.figure()
ax = plt.axes()
c = ax.pcolormesh(x,vmin=0,vmax=20,cmap=cm.cm.deep)
plt.colorbar(c,ax=ax,orientation='vertical', location='right')
plt.show()

fig = plt.figure()
ax = plt.axes()
c = ax.pcolormesh(y_good,vmin=0,vmax=20,cmap=cm.cm.deep)
plt.colorbar(c,ax=ax,orientation='vertical', location='right')
plt.show()

fig = plt.figure()
ax = plt.axes()
c = ax.pcolormesh(y_bad,vmin=0,vmax=20,cmap=cm.cm.deep)
plt.colorbar(c,ax=ax,orientation='vertical', location='right')
plt.show()
# %%
fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharex=True, sharey=True, gridspec_kw={'width_ratios': [1, 1, 1, 0.01]})
fig.suptitle("Example 2", fontsize=16)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

c = axes[0].pcolormesh(x,vmin=0,vmax=20,cmap=cm.cm.deep)
#plt.colorbar(c,ax=axes[0],orientation='vertical', location='right')
axes[0].set_title('Observations')

c = axes[1].pcolormesh(y_good,vmin=0,vmax=20,cmap=cm.cm.deep)
#plt.colorbar(c,ax=axes[1],orientation='vertical', location='right')
axes[1].set_title('Model A')

c = axes[2].pcolormesh(y_bad,vmin=0,vmax=20,cmap=cm.cm.deep)
axes[2].set_title('Model B')
plt.colorbar(c,ax=axes[3],orientation='vertical', location='right',fraction=4, pad=0.04)
axes[3].axis('off')

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(out_path,'Example2.pdf'), dpi=300)
plt.close()
# %%



for y in [y_good,y_bad]:   
    spaef_val, wspaef_val, mspaef_val, esp_val = calculate_spaef(x,y)
    print('WSPAEF: {}, SPAEF: {}, MSPAEF: {}, E_sp: {}'.format(wspaef_val,spaef_val,mspaef_val,esp_val))
# %%
# Generate SPAEF values for different parameter combinations
for i in range(20):
    x = generate_synthetic_field(dist_matrix, grid_size=grid_size, std_dev=1.0, mean=6.0, length_scale=1.0) 

    x = np.where(x>0,x,0)
    disturbed_field = generate_synthetic_field(dist_matrix, grid_size=grid_size, std_dev=1, mean=0.0, length_scale=1.0) 
    y_good = generate_correlated_field(x,target_corr=0.6,std_ratio=1.1,bias=0.0)
    y_good = np.where(y_good>0,y_good,0)
    y_bad = generate_correlated_field(x,target_corr=0.8,std_ratio=2.0,bias=7.5)
    y_bad = np.where(y_bad>0,y_bad,0)
    
    spaef_val, wspaef_val, mspaef_val, esp_val = calculate_spaef(x,y_good)
    spaef_results['Good'].append((spaef_val,wspaef_val,mspaef_val,esp_val))
    #print(mspaef_val,(1-np.exp(-nmse))**2,(1-np.exp(-wd_val/np.std(x)))**2,(1-np.exp(-np.abs(1-std_ratio)))**2,0.25*(1-corr_val)**2)
    
    spaef_val, wspaef_val, mspaef_val, esp_val = calculate_spaef(x,y_bad)
    spaef_results['Bad'].append((spaef_val,wspaef_val,mspaef_val,esp_val))
    
fig, axes = plt.subplots(1,4,figsize=(12,6))
for j, metric in enumerate(['WSPAEF', 'MSPAEF', 'SPAEF', '$E_{sp}$']):
    if metric == 'WSPAEF':
        good_results = [f[1] for f in spaef_results['Good']]
        bad_results = [f[1] for f in spaef_results['Bad']]
    elif metric == 'MSPAEF':
        good_results = [f[2] for f in spaef_results['Good']]
        bad_results = [f[2] for f in spaef_results['Bad']]
    elif metric == 'SPAEF':
        good_results = [f[0] for f in spaef_results['Good']]
        bad_results = [f[0] for f in spaef_results['Bad']]
    else:
        good_results = [f[3] for f in spaef_results['Good']]
        bad_results = [f[3] for f in spaef_results['Bad']]
    ax = axes[j]
    ax.boxplot(good_results, vert=True, positions=[1])
    ax.boxplot(bad_results, vert=True, positions=[2])
    ax.set_title(metric)
    ax.set_xticks([1,2],['Model A','Model B'], fontsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(out_path,'Example2_metrics.pdf'), dpi=300)
plt.close()