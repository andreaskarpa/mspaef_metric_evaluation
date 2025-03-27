#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:59:04 2024

@author: andreaskarpasitis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.special import kv, gamma
from scipy import stats
from sklearn import metrics
import warnings
import os
import matplotlib as mpl
from tqdm import tqdm

mpl.rcParams["font.family"] = "Arial" #"Ani"
warnings.filterwarnings("ignore")

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
def generate_synthetic_field(dist_matrix, grid_size=5, std=1.0, mean=0.0, length_scale=1.0):
    """
    Generate synthetic field x with normal distribution
    with specified mean and std
    
    Parameters:
    - dist_matrix: np.array, the distance matrix.
    - grid_size: float, the size of each of the dimensions of the matrix.
    - std: float, desired standard deviation ratio of x
    - mean: float, desired mean value of x.
    
    Returns:
    - field: np.array, synthetic field  
          with specified std and mean value.
    """
    cov_matrix = matern_covariance(dist_matrix, length_scale=length_scale)
    d = dist_matrix.shape[0]
    sigma_current = np.sqrt(np.trace(cov_matrix) / d)
    field = np.random.multivariate_normal(mean * np.ones(dist_matrix.shape[0]), (std/sigma_current) ** 2 * cov_matrix) + 0
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
    - x: 1D np.array, the flattened array of comparison dataset.
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
    
    corr = np.corrcoef(x.flatten(), y.flatten())[0, 1] #### Pearson Correlation coefficient, for use in SPAEF, MSPAEF and WSPAEF metrics
    rs = stats.spearmanr(x.flatten(),y.flatten())[0] #### Spearman Correlation coefficient, for use in Esp metric
    
    q75, q25 = np.percentile(x, [75 ,25])
    iqr_x = q75 - q25

    nmse = np.sqrt(np.sum((x-y)**2)/(len(x.flatten())))/iqr_x  #### Normalized (over std of comparison dataset) Root Mean Square Error of original data  
    wd_data = stats.wasserstein_distance(bin_edges[:-1],bin_edges[:-1], data_comp_pdf, data_pdf )  ##### Wasserstein distance of histograms of the original data
    #wd_data = stats.wasserstein_distance(bin_edges[:-1],bin_edges[:-1], data_comp_mean_pdf, data_mean_pdf )  ##### Wasserstein distance of histograms of the original data
    alpha = 1 - metrics.root_mean_squared_error(zscore_data_comp,zscore_data)   ### Root Mean Square Error of z-score data, for use in Esp
    
    spaef = np.sqrt((1 - b) ** 2 + (1 - gamma_data_zscore) ** 2 + (1 - corr) ** 2)   ### SPAEF metric, modified so that 0 indicates perfect model preformance 
    wspaef = np.sqrt( (1 - corr) ** 2 + (1 - 1/std_ratio) ** 2 + wd_data**2)         ### WSPAEF metric
    mspaef = np.sqrt((nmse)**2 + (np.sqrt((std_ratio**2 - 1)**2 + (1/std_ratio**2 - 1)**2))**2 + ((np.abs(np.mean(x)-np.mean(y))/iqr_x))**2 + (1-corr)**2)/np.sqrt(4) ### MSPAEF metric, modified so that 0 indicates perfect model preformance
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
    y_normal = y_normal* lambda_ratio #+ mean_x
    
    # Step 5: Apply the log-normal transformation to obtain the skewed field y
    y = np.exp(y_normal)

    # Step 6: Add the desired bias to y in log-normal space
    y += delta_bias
    
    y = np.where(y>0,y,0)
    
    return y


out_path = '/path/to/somewhere/' ## Path to output the figures
# Parameters
num_realizations = 200  ### Number of examples to use
grid_size = 10          ### Size of grids to use for each example
correlations = [-0.9, -0.3, 0.3, 0.9]  # Correlation values
std_ratios = [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9]  # Variation ratios
std_ratios = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8]  # Variation ratios
biases = [0.0, 1.0, 3.0]  # Bias values


dict_axes_lim_wspaef = {0.0:[0,3], 1.0:[0,3], 3.0:[2,5]} #### Y range for WSPAEF plot 
dict_axes_lim_spaef = {0.0:[0,3], 1.0:[0,3], 3.0:[0,3]} #### Y range for SPAEF and Esp plot
dict_axes_lim_mspaef = {0.0:[0,4], 1.0:[0,4], 3.0:[1,5]} #### Y range for MSPAEF plot


# Distance matrix for Matern covariance
dist_matrix = create_distance_matrix(grid_size)

# Containers to store results
spaef_results = {"Undisturbed": [], "Disturbed": []}

# Generate SPAEF values for different parameter combinations
for corr in tqdm(correlations, desc='Correlations'):
    for std_ratio in tqdm(std_ratios, desc='Std ratios', leave=False):
        for bias in tqdm(biases, desc = 'biases', leave=False):
            undisturbed_spaefs = []
            disturbed_spaefs = []
            undisturbed_mspaefs = []
            disturbed_mspaefs = []
            undisturbed_wspaefs = []
            disturbed_wspaefs = []
            undisturbed_esp = []
            disturbed_esp = []
            
            for _ in tqdm(range(num_realizations),desc='Realizations', leave=False):
                # Generate undisturbed field x
                x = generate_synthetic_field(dist_matrix, grid_size=grid_size, std=1.0, mean=10.0, length_scale=1.0) 
                # Generate undisturbed correlated field y
                y_undisturbed = generate_correlated_field(x, target_corr=corr, std_ratio=std_ratio, bias=bias)
                # Calculate SPAEF for undisturbed case
                spaef_val, wspaef_val, mspaef_val, esp_val = calculate_spaef(x, y_undisturbed)
                undisturbed_spaefs.append(spaef_val)
                undisturbed_wspaefs.append(wspaef_val)
                undisturbed_mspaefs.append(mspaef_val)
                undisturbed_esp.append(esp_val)

                # Generate disturbed field y (log-normal transformation)
                x_disturbed = np.exp(x - np.mean(x))
                y_disturbed = generate_skewed_correlated_field(x_disturbed, corr,lambda_ratio=std_ratio,delta_bias=bias)
                
                # Calculate SPAEF for disturbed case
                spaef_val, wspaef_val, mspaef_val, esp_val = calculate_spaef(x_disturbed, y_disturbed)
                disturbed_spaefs.append(spaef_val)
                disturbed_wspaefs.append(wspaef_val)
                disturbed_mspaefs.append(mspaef_val)
                disturbed_esp.append(esp_val)

                
            # Store mean SPAEF for plotting
            spaef_results["Undisturbed"].append((corr, std_ratio, bias, np.median(undisturbed_spaefs), np.median(undisturbed_wspaefs), np.median(undisturbed_mspaefs), np.median(undisturbed_esp)))
            spaef_results["Disturbed"].append((corr, std_ratio, bias, np.median(disturbed_spaefs),np.median(disturbed_wspaefs), np.median(disturbed_mspaefs), np.median(disturbed_esp)))



# Plotting SPAEF behavior
fig, axes = plt.subplots(len(biases), len(correlations), figsize=(15, 10), sharex=True, sharey=False)
fig.suptitle("SPAEF Behavior", fontsize=21)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Loop through each subplot for a combination of correlation and bias
for i, bias in enumerate(biases):
    for j, corr in enumerate(correlations):
        # Filter the SPAEF results for current correlation and bias
        undisturbed = [val[3] for val in spaef_results["Undisturbed"] if val[0] == corr and val[2] == bias]
        disturbed = [val[3] for val in spaef_results["Disturbed"] if val[0] == corr and val[2] == bias]
        
        # X-axis is the standard deviation ratio
        std_ratio_vals = std_ratios

        # Plot on the respective subplot
        ax = axes[i, j]
        ax.plot(std_ratio_vals, undisturbed, 'o-', label="normal", color="purple")
        ax.plot(std_ratio_vals, disturbed, '^-', label="skewed", color="teal")
        
        # Set titles and labels
        if i == 0:
            ax.set_title(f"$\\rho$ = {corr}",fontsize=20)
        if j == 0:
            ax.set_ylabel(f"$\\delta$ = {bias}",fontsize=20)
    
        ax.set_xlabel("$\\lambda$",fontsize=20)
        ax.set_ylim(dict_axes_lim_spaef[bias][0], dict_axes_lim_spaef[bias][1])  # Adjust based on observed SPAEF range
        ax.tick_params(axis='y', which='major', labelsize=18)
        ax.set_xticks([0.5,1.0,1.5],[0.5,1.0,1.5],fontsize=18)
        ax.grid()
# Add a legend outside the plot
handles, labels = ax.get_legend_handles_labels()
axes[0,j].legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(out_path,'spaef_behavior.pdf'), dpi=300)
plt.close()


# Plotting Esp behavior
fig, axes = plt.subplots(len(biases), len(correlations), figsize=(15, 10), sharex=True, sharey=False)
fig.suptitle("$E_{sp}$ Behavior", fontsize=21)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Loop through each subplot for a combination of correlation and bias
for i, bias in enumerate(biases):
    for j, corr in enumerate(correlations):
        # Filter the SPAEF results for current correlation and bias
        undisturbed = [val[6] for val in spaef_results["Undisturbed"] if val[0] == corr and val[2] == bias]
        disturbed = [val[6] for val in spaef_results["Disturbed"] if val[0] == corr and val[2] == bias]
        
        # X-axis is the standard deviation ratio
        std_ratio_vals = std_ratios

        # Plot on the respective subplot
        ax = axes[i, j]
        ax.plot(std_ratio_vals, undisturbed, 'o-', label="normal", color="purple")
        ax.plot(std_ratio_vals, disturbed, '^-', label="skewed", color="teal")
        
        # Set titles and labels
        if i == 0:
            ax.set_title(f"$\\rho$ = {corr}",fontsize=20)
        if j == 0:
            ax.set_ylabel(f"$\\delta$ = {bias}",fontsize=20)
    
        ax.set_xlabel("$\\lambda$",fontsize=20)
        ax.set_ylim(dict_axes_lim_spaef[bias][0], dict_axes_lim_spaef[bias][1])  # Adjust based on observed SPAEF range
        ax.tick_params(axis='y', which='major', labelsize=18)
        ax.set_xticks([0.5,1.0,1.5],[0.5,1.0,1.5],fontsize=18)
        ax.grid()
# Add a legend outside the plot
handles, labels = ax.get_legend_handles_labels()
axes[0,j].legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(out_path,'esp_behavior.pdf'), dpi=300)
plt.close()


# Plotting WSPAEF behavior
fig, axes = plt.subplots(len(biases), len(correlations), figsize=(15, 10), sharex=True, sharey=False)
fig.suptitle("WSPAEF Behavior", fontsize=21)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
# Loop through each subplot for a combination of correlation and bias
for i, bias in enumerate(biases):
    for j, corr in enumerate(correlations):
        # Filter the SPAEF results for current correlation and bias
        undisturbed = [val[4] for val in spaef_results["Undisturbed"] if val[0] == corr and val[2] == bias]
        disturbed = [val[4] for val in spaef_results["Disturbed"] if val[0] == corr and val[2] == bias]
        
        # X-axis is the standard deviation ratio
        std_ratio_vals = std_ratios

        # Plot on the respective subplot
        ax = axes[i, j]
        ax.plot(std_ratio_vals, undisturbed, 'o-', label="normal", color="purple")
        ax.plot(std_ratio_vals, disturbed, '^-', label="skewed", color="teal")
        #ax.set_yscale('log')
        # Set titles and labels
        if i == 0:
            ax.set_title(f"$\\rho$ = {corr}",fontsize=20)
        if j == 0:
            ax.set_ylabel(f"$\\delta$ = {bias}",fontsize=20)
    
        ax.set_xlabel("$\\lambda$",fontsize=20)
        ax.set_ylim(dict_axes_lim_wspaef[bias][0], dict_axes_lim_wspaef[bias][1])  # Adjust based on observed SPAEF range
        ax.tick_params(axis='y', which='major', labelsize=18)
        ax.set_xticks([0.5,1.0,1.5],[0.5,1.0,1.5],fontsize=18)
        ax.grid()
# Add a legend outside the plot
handles, labels = ax.get_legend_handles_labels()
axes[0,j].legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(out_path,'wspaef_behavior.pdf'), dpi=300)
plt.close()


# Plotting MSPAEF behavior
fig, axes = plt.subplots(len(biases), len(correlations), figsize=(15, 10), sharex=True, sharey=False)
fig.suptitle("MSPAEF Behavior", fontsize=21)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
# Loop through each subplot for a combination of correlation and bias
for i, bias in enumerate(biases):
    for j, corr in enumerate(correlations):
        # Filter the SPAEF results for current correlation and bias
        undisturbed = [val[5] for val in spaef_results["Undisturbed"] if val[0] == corr and val[2] == bias]
        disturbed = [val[5] for val in spaef_results["Disturbed"] if val[0] == corr and val[2] == bias]
        
        # X-axis is the standard deviation ratio
        std_ratio_vals = std_ratios

        # Plot on the respective subplot
        ax = axes[i, j]
        ax.plot(std_ratio_vals, undisturbed, 'o-', label="normal", color="purple")
        ax.plot(std_ratio_vals, disturbed, '^-', label="skewed", color="teal")
        #ax.set_yscale('log')
        # Set titles and labels
        if i == 0:
            ax.set_title(f"$\\rho$ = {corr}",fontsize=20)
        if j == 0:
            ax.set_ylabel(f"$\\delta$ = {bias}",fontsize=20)
    
        ax.set_xlabel("$\\lambda$",fontsize=20)
        ax.set_ylim(dict_axes_lim_mspaef[bias][0], dict_axes_lim_mspaef[bias][1])  # Adjust based on observed SPAEF range
        ax.tick_params(axis='y', which='major', labelsize=18)
        ax.set_xticks([0.5,1.0,1.5],[0.5,1.0,1.5],fontsize=18)
        ax.grid()
# Add a legend outside the plot
handles, labels = ax.get_legend_handles_labels()
axes[0,j].legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(out_path,'mspaef_behavior.pdf'), dpi=300)
plt.close()