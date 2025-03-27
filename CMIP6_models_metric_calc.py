#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:47:53 2025

@author: andreaskarpasitis
"""

import numpy as np
import xarray as xr
import os
import datetime as dt
import copy
import pandas as pd
from scipy import stats
from datetime import datetime as dt
from sklearn import metrics
from xarray_regrid.methods.conservative import conservative_regrid


def convert_to_dt(x):
    try:
        return dt.strptime(str(x), '%Y-%m-%d %H:%M:%S')
    except:
        return dt.strptime(str(x).split(".")[0], '%Y-%m-%d %H:%M:%S')


def convert_array_to_dt(array):
    return [convert_to_dt(x) for x in array]

def change_pr_variable_name(ncfile):
    vars_exist = list(ncfile.data_vars)
    if 'pr' not in vars_exist:
        if 'tp' in vars_exist:
            ncfile = ncfile.rename({'tp':'pr'})
    return ncfile

def remove_nans(x, y):
    """
    Removes NaN values from both x and y, ensuring only non-NaN values remain.
    
    Parameters:
    x (np.ndarray): First input array.
    y (np.ndarray): Second input array.
    
    Returns:
    np.ndarray, np.ndarray: Filtered arrays with only non-NaN values.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)  # Keep only elements that are non-NaN in both arrays
    return x[mask], y[mask]

def renaming_dimensions(ncfile):
    variables = ncfile.dims
    print(variables)
    if 'latitude' in variables:
        ncfile = ncfile.rename({'latitude':'lat'})
    if 'longitude' in variables:
        ncfile = ncfile.rename({'longitude':'lon'})
    if 'nav_lat' in variables:
        ncfile = ncfile.rename({'nav_lat':'lat'})
    if 'nav_lon' in variables:
        ncfile = ncfile.rename({'nav_lon':'lon'})
    if 'time_counter' in variables:
        ncfile = ncfile.rename({'time_counter':'time'})
    if 'valid_time' in variables:
        ncfile = ncfile.rename({'valid_time':'time'})
    if 'plev' in variables:
        ncfile = ncfile.rename({'plev':'level'})
    elif 'pressure' in variables:
        ncfile = ncfile.rename({'pressure':'level'})
    elif 'pressure_level' in variables:
        ncfile = ncfile.rename({'pressure_level':'level'})
    if 'date' in variables: 
        ncfile = ncfile.rename({'date':'time'})
        ncfile['time'] = pd.to_datetime(ncfile['time'].astype(str), format='%Y%m%d')
    if 'expver' and 'number' in ncfile.coords:
        ncfile = ncfile.drop_vars(['expver', 'number'])   
    return ncfile

def convert_lon_to_360(da):
    """
    Convert longitudes from the range [-180, 180] to [0, 360].

    Parameters:
    da (xr.DataArray or xr.Dataset): Input data with longitude dimension.

    Returns:
    xr.DataArray or xr.Dataset: Data with longitudes converted to [0, 360].
    """
    if da['lon'].min() < 0:
        print('Chaning coordinates from [-180,180] to [0,360]')
        da = da.assign_coords(lon=(da['lon'] % 360))
        da = da.sortby('lon')
    return da

def add_longitudes_circular(da):
    print('Extending longitudes for interpolation if needed')
    lons = da['lon'].values
    if 360 not in lons:
        print('Extending near 360')
        # Extract the data corresponding to the closest at lon=0
        data_at_near_zero_lon = da.sel(lon=0, method='nearest')    
        # Create a new DataArray with lon=360, copying the data from lon=0
        new_data_at_near360 = data_at_near_zero_lon.expand_dims({'lon': [360+np.min(lons)]}, axis=-1)
        da = xr.concat([da , new_data_at_near360], dim='lon')
        da = da.sortby('lon')
    if 0 not in lons:
        print('Extending near 0')
        # Extract the data corresponding to the closest at lon=360
        data_at_near_360_lon = da.sel(lon=360, method='nearest')
        new_data_at_near0 = data_at_near_360_lon.expand_dims({'lon': [np.max(lons)-360]}, axis=-1)
        # Combine the new data with the original DataArray
        da = xr.concat([new_data_at_near0 ,da], dim='lon')
        
        # Sort the longitude dimension if needed (optional)
        da = da.sortby('lon')
    return da

def add_poles_latitudes(da):
    print('Adding latitudes of +-90 for interpolation if needed')
    lats = da['lat'].values
    if 90 not in lats:
        print('Adding 90 latitude')
        # Extract the data corresponding to the closest at lon=0
        data_at_near_north_pole = da.sel(lat=90, method='nearest')    
        # Create a new DataArray with lon=360, copying the data from lon=0
        new_data_at_north_pole = data_at_near_north_pole.expand_dims({'lat': [90]}, axis=-1)
        da = xr.concat([da , new_data_at_north_pole], dim='lat')
        da = da.sortby('lat')
    if -90 not in lats:
        print('Adding -90 latitude')
        # Extract the data corresponding to the closest at lon=360
        data_at_near_south_pole = da.sel(lat=-90, method='nearest')
        new_data_at_near_south_pole = data_at_near_south_pole.expand_dims({'lat': [-90]}, axis=-1)
        # Combine the new data with the original DataArray
        da = xr.concat([new_data_at_near_south_pole ,da], dim='lat')
        
        # Sort the longitude dimension if needed (optional)
        da = da.sortby('lat')
    return da

def remapping(array_var,dict_lonlat):
    lons = np.linspace(0, 360, dict_lonlat['lon'])
    lats = np.linspace(-90, 90, dict_lonlat['lat'])
    print('Remapping data from {}x{} to {}x{} resolution'.format(len(array_var.lon),len(array_var.lat),len(lons),len(lats)))
    array_var = array_var.interp(lon=lons,lat=lats,method='linear')
    return array_var

def directly_read_prepare_pr_dataarrays(path,dict_lonlat,var_name,first_year,last_year, path_temp, method, rate_to_daily):
    dataarray = xr.open_dataset(path)
    dataarray = change_pr_variable_name(dataarray)
    dataarray = dataarray[var_name]
    dataarray = dataarray.where((dataarray.time.dt.year>=first_year)&(dataarray.time.dt.year<=last_year), drop=True)
    if rate_to_daily:
        dataarray = dataarray*86400
    dataarray = renaming_dimensions(dataarray)
    dataarray = convert_lon_to_360(dataarray)
    if method =='conservative':
        dataarray = conservative_remapping(dataarray, dict_lonlat, path_temp, var_name, 36)
    elif method == 'linear':
        dataarray = add_longitudes_circular(dataarray)
        dataarray = add_poles_latitudes(dataarray)
        dataarray = remapping(dataarray, dict_lonlat)
    else:
        print('Incorrect remapping method requested!')
    dataarray = dataarray.where((dataarray.time.dt.year>=first_year)&(dataarray.time.dt.year<=last_year),drop=True)
    return dataarray

def directly_read_prepare_var_dataarrays(path,dict_lonlat,var_name,first_year,last_year, path_temp, method,mul_const=1):
    dataarray = xr.open_dataset(path)
    dataarray = dataarray[var_name]   
    dataarray = dataarray.where((dataarray.time.dt.year>=first_year)&(dataarray.time.dt.year<=last_year), drop=True)
    dataarray = renaming_dimensions(dataarray)
    dataarray = convert_lon_to_360(dataarray)
    if method =='conservative':
        dataarray = conservative_remapping(dataarray, dict_lonlat, path_temp, var_name, 36)
    elif method == 'linear':
        dataarray = add_longitudes_circular(dataarray)
        dataarray = add_poles_latitudes(dataarray)
        dataarray = remapping(dataarray, dict_lonlat)
    else:
        print('Incorrect remapping method requested!')
    dataarray = dataarray.where((dataarray.time.dt.year>=first_year)&(dataarray.time.dt.year<=last_year),drop=True)
    dataarray = dataarray.transpose('time', 'lat', 'lon')
    if mul_const != 1:
        dataarray = dataarray*mul_const  ### From flux per second, to daily averaged precipitation
    return dataarray

def conservative_remapping(array_var,dict_lonlat, path, var, n):
    lons = np.linspace(0, 360, dict_lonlat['lon'])
    lats = np.linspace(-90, 90, dict_lonlat['lat'])
    print('Conservative Remapping data from {}x{} to {}x{} resolution'.format(len(array_var.lon),len(array_var.lat),len(lons),len(lats)))
    grid_tgt = xr.Dataset(coords={"lat": (["lat"], lats), "lon": (["lon"], lons)})
    
    time_len = len(array_var['time'])  # Total number of time steps
    
    # Split into chunks of size n
    time_groups = [slice(i, min(i + n, time_len)) for i in range(0, time_len, n)]

    
    # Regrid each time step separately and save results incrementally
    output_files = []
    for i, time_slice in enumerate(time_groups):
    
        # Select one time step
        ds_t = array_var.isel(time=time_slice)
        print(f"Processing time step {ds_t.time.values[0]} to {ds_t.time.values[-1]}...")
        # Apply conservative regridding (lazy computation)
        ds_regridded = conservative_regrid(
            data=ds_t,
            target_ds=grid_tgt,
            latitude_coord="lat",
            skipna=True,
            nan_threshold=1.0
        )
    
        # Save each time step separately
        output_filename = f"regridded_{i}.nc"
        # Convert the sparse array to dense before saving
        ds_regridded.data = ds_regridded.data.todense()
        ds_regridded.to_netcdf(os.path.join(path,output_filename))
    
        # Store filename for merging later
        output_files.append(output_filename)
    
    print("Regridding complete. Merging all files...")
    
    # Merge all saved files into a single dataset
    ds_final = xr.open_mfdataset([os.path.join(path,f) for f in output_files], combine="nested", concat_dim="time")
    for file in output_files:
        try:
            os.remove(os.path.join(path,file))
            print(f"Successfully removed: {file}")
        except OSError as e:
            print(f"Error removing file {file}: {e}")
    return ds_final[var]


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
    x, y = remove_nans(x,y)
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
    alpha = 1 - metrics.root_mean_squared_error(zscore_data_comp,zscore_data)   ### Root Mean Square Error of z-score data, for use in Esp
    
    spaef = 1 - np.sqrt((1 - b) ** 2 + (1 - gamma_data_zscore) ** 2 + (1 - corr) ** 2)   ### SPAEF metric, modified so that 0 indicates perfect model preformance 
    wspaef = np.sqrt( (1 - corr) ** 2 + (1 - 1/std_ratio) ** 2 + wd_data**2)         ### WSPAEF metric
  
    mspaef = 1 - np.sqrt((nmse)**2 + (np.sqrt((std_ratio**2 - 1)**2 + (1/std_ratio**2 - 1)**2))**2 + ((np.abs(np.mean(x)-np.mean(y))/iqr_x))**2 + (1-corr)**2)/np.sqrt(4)  ### MSPAEF metric, modified so that 0 indicates perfect model preformance
    
    
    esp = 1 - np.sqrt((rs-1)**2 + (b-1)**2 + (alpha - 1)**2)                             ### Esp metric, modified so that 0 indicates perfect model preformance 
    
    return spaef,wspaef, mspaef, esp, corr, std_ratio, np.mean(x)-np.mean(y), nmse

def r_squared(y_true, y_pred):
    """
    Calculates the R-squared score between two 2D arrays.

    Args:
        y_true: The true values.
        y_pred: The predicted values.

    Returns:
        The R-squared score.
    """

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def evaluation_precipitation(data_model_o,data_obs_o,first_year,last_year,out_path,name):   
    season = [1,2,3,4,5,6,7,8,9,10,11,12]
    print('Doing model {}'.format(name))

    columns = ['Model','MSPAEF','SPAEF','WSPAEF', 'Esp', 'Corr', 'Std ratio', 'Bias', 'NRMSE']
    
    latitude_range = [-90,90]
    data_obs_tropical = data_obs_o.where((data_obs_o.time.dt.year >= first_year)&(data_obs_o.time.dt.year <= last_year)&(data_obs_o.lat>=latitude_range[0])&(data_obs_o.lat<=latitude_range[1]), drop=True)
    data_model_tropical = data_model_o.where((data_model_o.time.dt.year >= first_year)&(data_model_o.time.dt.year <= last_year)&(data_model_o.lat>=latitude_range[0])&(data_model_o.lat<=latitude_range[1]), drop=True)
    
    data_obs = copy.deepcopy(data_obs_tropical)
    data_model = copy.deepcopy(data_model_tropical)
    
    data_obs = data_obs * data_obs.time.dt.daysinmonth
    data_model = data_model * data_model.time.dt.daysinmonth
    
    #bins = np.logspace(np.log10(0.01), np.log10(100), num=20)
    
    data_obs = data_obs.isel(time=data_obs.time.dt.month.isin(season),drop=True).groupby('time.year').sum(dim='time')
    data_model = data_model.isel(time=data_model.time.dt.month.isin(season),drop=True).groupby('time.year').sum(dim='time')
    
    data_obs_sea_mean = data_obs.mean(dim='year').values.flatten(order='F')
    data_model_sea_mean = data_model.mean(dim='year').values.flatten(order='F')
    
    spaef_val ,wspaef_val, mspaef_val, esp_val, corr, std_ratio, bias, nrmse = calculate_spaef(data_obs_sea_mean,data_model_sea_mean)
    
    data_df = [[name,np.around(mspaef_val,3),np.around(spaef_val,3),np.around(wspaef_val,3),np.around(esp_val,3), np.around(corr,3), np.around(std_ratio,3), np.around(bias, 2), np.around(nrmse, 3)]]
    df_model = pd.DataFrame(data=data_df,columns=columns)
    
    return df_model

def evaluation_temperature(data_model_o,data_obs_o,first_year,last_year,out_path,name):   
    season = [1,2,3,4,5,6,7,8,9,10,11,12]
    print('Doing model {}'.format(name))

    columns = ['Model','MSPAEF','SPAEF','WSPAEF', 'Esp', 'Corr', 'Std ratio', 'Bias', 'NRMSE']
    
    latitude_range = [-90,90]
    data_obs_tropical = data_obs_o.where((data_obs_o.time.dt.year >= first_year)&(data_obs_o.time.dt.year <= last_year)&(data_obs_o.lat>=latitude_range[0])&(data_obs_o.lat<=latitude_range[1]), drop=True)
    data_model_tropical = data_model_o.where((data_model_o.time.dt.year >= first_year)&(data_model_o.time.dt.year <= last_year)&(data_model_o.lat>=latitude_range[0])&(data_model_o.lat<=latitude_range[1]), drop=True)
    
    data_obs = copy.deepcopy(data_obs_tropical)
    data_model = copy.deepcopy(data_model_tropical)
    
    #bins = np.logspace(np.log10(0.01), np.log10(100), num=20)
    
        
    data_obs_sea_mean = data_obs.isel(time=data_obs.time.dt.month.isin(season),drop=True).mean(dim='time').values.flatten(order='F')
    data_model_sea_mean = data_model.isel(time=data_model.time.dt.month.isin(season),drop=True).mean(dim='time').values.flatten(order='F')
    
    spaef_val ,wspaef_val, mspaef_val, esp_val, corr, std_ratio, bias, nrmse = calculate_spaef(data_obs_sea_mean,data_model_sea_mean)
    
    data_df = [[name,np.around(mspaef_val,3),np.around(spaef_val,3),np.around(wspaef_val,3),np.around(esp_val,3), np.around(corr,3), np.around(std_ratio,3), np.around(bias, 2), np.around(nrmse, 3)]]
    df_model = pd.DataFrame(data=data_df,columns=columns)
    
    return df_model


if __name__ == "__main__":
    first_year = 1981
    last_year = 2010
    lons=360
    lats=180
    var_name= 'tas'       #### choose either 'tas' for temperature variable or 'pr' for precipitation variable
    method_pr = 'conservative'
    method_tas = 'linear'
    
    out_path = '/path/to/somewhere/'   ### path where to output the metrics values
    path = '/path/to/somewhere/'       ### path where the CMIP6 model data reside (nc files)
    
    path_comp_pr_direct = '/path/to/somefile.nc'    ## path to comparison precipitation file
    path_comp_tas_direct = '/path/to/somefile.nc'   ## path to comparison temperature file
    
    path_temp = '/path/to/somewhere/'      ### path to write temporary files for conservative remapping
    
    name_model = '{}_mon_mod_ssp245_192_0{}.nc'   ## generalized form of CMIP6 file names download from Climate Explorer

    
    out_name = 'Metrics_CMIP6_models_{}.csv'    ### Generalized form of output csv files
    total_df = pd.DataFrame()

    dict_lonlat = {'lon':lons, 
                  'lat':lats}
    print('Calculating reference dataset')
    if var_name == 'pr':
        rate_to_daily = False
        ncfile_pr_comp = directly_read_prepare_pr_dataarrays(path_comp_pr_direct, dict_lonlat, var_name, first_year, last_year, path_temp, method_pr, rate_to_daily)
    else:
        ncfile_tas_comp = directly_read_prepare_var_dataarrays(path_comp_tas_direct, dict_lonlat, 't2m', first_year, last_year, path_temp, method_tas)
    
    for i in range(0,33):
        print('Calculating model {}'.format(i))
        if i<10:
            n_i = '0{}'.format(i)
        elif i<100:
            n_i = '{}'.format(i)
        file_model = os.path.join(path,name_model.format(var_name,n_i))
        
        if var_name == 'pr':
            rate_to_daily = True
            ncfile_pr_model = directly_read_prepare_pr_dataarrays(file_model, dict_lonlat, var_name, first_year, last_year, path_temp, method_pr, rate_to_daily)
            df_model = evaluation_precipitation(ncfile_pr_model,ncfile_pr_comp,first_year,last_year,out_path,i)
        else:
            ncfile_tas_model = directly_read_prepare_var_dataarrays(file_model, dict_lonlat, var_name, first_year, last_year, path_temp, method_tas)
            df_model = evaluation_temperature(ncfile_tas_model,ncfile_tas_comp,first_year,last_year,out_path,i)
        total_df = pd.concat([total_df,df_model])
    total_df.to_csv(os.path.join(out_path,out_name.format(var_name)))