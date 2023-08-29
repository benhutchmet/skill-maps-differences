# Only the plotting functions are defined here

# Imports
import argparse
import os
import sys
import glob
import re

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from datetime import datetime
import scipy.stats as stats
import matplotlib.animation as animation
from matplotlib import rcParams
from PIL import Image
import multiprocessing
import dask.array as da
import dask.distributed as dd
import dask

# Import CDO
from cdo import *
cdo = Cdo()

# Import the dictionaries
import dictionaries as dic
import functions as fnc

# Plot the correlation coefficients and p-values for the multi-model mean
def plot_correlations(model, rfield, pfield, obs, variable, region, season, forecast_range, plots_dir, obs_lons_converted, lons_converted, azores_grid, iceland_grid, uk_n_box, uk_s_box, p_sig=0.05, experiment=None, observed_data=None, ensemble_members_count=None):
    """Plot the correlation coefficients and p-values.
    
    This function plots the correlation coefficients and p-values
    for a given variable, region, season and forecast range.
    
    Parameters
    ----------
    model : str
        Name of the model.
    rfield : array
        Array of correlation coefficients.
    pfield : array
        Array of p-values.
    obs : str
        Observed dataset.
    variable : str
        Variable.
    region : str
        Region.
    season : str
        Season.
    forecast_range : str
        Forecast range.
    plots_dir : str
        Path to the directory where the plots will be saved.
    obs_lons_converted : array
        Array of longitudes for the observed data.
    lons_converted : array
        Array of longitudes for the model data.
    azores_grid : array
        Array of longitudes and latitudes for the Azores region.
    iceland_grid : array
        Array of longitudes and latitudes for the Iceland region.
    uk_n_box : array
        Array of longitudes and latitudes for the northern UK index box.
    uk_s_box : array
        Array of longitudes and latitudes for the southern UK index box.
    p_sig : float
        P-value for statistical significance. Default is 0.05.
    experiment : str
        Experiment. Default is None.
    observed_data : xarray.Dataset
        Observed data. Default is None.
    ensemble_members_count : dict
        Dictionary of the number of ensemble members for each model. Default is None.
    """

    # Extract the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid['lon1'], azores_grid['lon2']
    azores_lat1, azores_lat2 = azores_grid['lat1'], azores_grid['lat2']

    # Extract the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid['lon1'], iceland_grid['lon2']
    iceland_lat1, iceland_lat2 = iceland_grid['lat1'], iceland_grid['lat2']

    # Extract the lats and lons for the northern UK index box
    uk_n_lon1, uk_n_lon2 = uk_n_box['lon1'], uk_n_box['lon2']
    uk_n_lat1, uk_n_lat2 = uk_n_box['lat1'], uk_n_box['lat2']

    # Extract the lats and lons for the southern UK index box
    uk_s_lon1, uk_s_lon2 = uk_s_box['lon1'], uk_s_box['lon2']
    uk_s_lat1, uk_s_lat2 = uk_s_box['lat1'], uk_s_box['lat2']

    # subtract 180 from all of the azores and iceland lons
    azores_lon1, azores_lon2 = azores_lon1 - 180, azores_lon2 - 180
    iceland_lon1, iceland_lon2 = iceland_lon1 - 180, iceland_lon2 - 180

    # subtract 180 from all of the uk lons
    uk_n_lon1, uk_n_lon2 = uk_n_lon1 - 180, uk_n_lon2 - 180
    uk_s_lon1, uk_s_lon2 = uk_s_lon1 - 180, uk_s_lon2 - 180

    # set up the converted lons
    # Set up the converted lons
    lons_converted = lons_converted - 180

    # Set up the lats and lons
    # if the region is global
    if region == 'global':
        lats = obs.lat
        lons = lons_converted
    # if the region is not global
    elif region == 'north-atlantic':
        lats = obs.lat
        lons = lons_converted
    else:
        print("Error: region not found")
        sys.exit()

    # Set the font size for the plots
    plt.rcParams.update({'font.size': 12})

    # Set the figure size
    plt.figure(figsize=(10, 8))

    # Set the projection
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add coastlines
    ax.coastlines()

    # Add gridlines with labels for the latitude and longitude
    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    # gl.top_labels = False
    # gl.right_labels = False
    # gl.xlabel_style = {'size': 12}
    # gl.ylabel_style = {'size': 12}

    # Add green lines outlining the Azores and Iceland grids
    ax.plot([azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1], [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())
    ax.plot([iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1], [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())

    # Add green lines outlining the northern and southern UK index boxes
    ax.plot([uk_n_lon1, uk_n_lon2, uk_n_lon2, uk_n_lon1, uk_n_lon1], [uk_n_lat1, uk_n_lat1, uk_n_lat2, uk_n_lat2, uk_n_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())
    ax.plot([uk_s_lon1, uk_s_lon2, uk_s_lon2, uk_s_lon1, uk_s_lon1], [uk_s_lat1, uk_s_lat1, uk_s_lat2, uk_s_lat2, uk_s_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())

    # Add filled contours
    # Contour levels
    clevs = np.arange(-1.8, 1.8, 0.1)
    # Contour levels for p-values
    clevs_p = np.arange(0, 1.1, 0.1)
    # Plot the filled contours
    cf = plt.contourf(lons, lats, rfield, clevs, cmap='RdBu_r', transform=ccrs.PlateCarree())

    # replace values in pfield that are greater than 0.05 with nan
    pfield[pfield > p_sig] = np.nan

    # print the pfield
    # print("pfield mod", pfield)

    # Add stippling where rfield is significantly different from zero
    plt.contourf(lons, lats, pfield, hatches=['....'], alpha=0, transform=ccrs.PlateCarree())

    # Add colorbar
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('Correlation Coefficient')

    # extract the model name from the list
    # given as ['model']
    # we only want the model name
    # if the length of the list is 1
    # then the model name is the first element
    if len(model) == 1:
        model = model[0]
    elif len(model) > 1:
        model = "multi-model mean"
    else :
        print("Error: model name not found")
        sys.exit()

    # if observed_data is not None:
    #     # Extract the first and last years
    if observed_data is not None:
        first_year = observed_data.time.dt.year.values[0]
        last_year = observed_data.time.dt.year.values[-1]
    else:
        first_year = None
        last_year = None

    # Set up the significance threshold
    # if p_sig is 0.05, then sig_threshold is 95%
    sig_threshold = int((1 - p_sig) * 100)

    # If ensemble_members_count is not None
    if ensemble_members_count is not None:
        total_no_members = sum(ensemble_members_count.values())
    else:
        total_no_members = None

    # Include the experiment in the title if it is not None
    if experiment is not None:
        # Add title
        plt.title(f"{model} {variable} {region} {season} {forecast_range} {experiment} {first_year}-{last_year} correlation coefficients, p < {p_sig} ({sig_threshold}%), N = {total_no_members}", fontsize=10)

        # Set up the figure name
        fig_name = f"{model}_{variable}_{region}_{season}_{forecast_range}_{experiment}_{total_no_members}_{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    else:
        # Add title
        plt.title(f"{model} {variable} {region} {season} {forecast_range} {first_year}-{last_year} correlation coefficients, p < {p_sig} ({sig_threshold}%), N = {total_no_members}", fontsize=10)

        # Set up the figure name
        fig_name = f"{model}_{variable}_{region}_{season}_{forecast_range}_{total_no_members}_{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # Show the figure
    plt.show()

# TODO: Include the bootstrapped p-values in the plot
# Define a function to plot subplots of the correlation coefficients and p-values for the init vs uninitialized models
def plot_correlations_init_vs_uninit(observed_data, init_model_data, uninit_model_data, init_models, uninit_models, variable, region, season, forecast_range, plots_dir, azores_grid, iceland_grid, uk_n_box, uk_s_box, p_sig=0.05):
    """Plot the correlation coefficients and p-values for the init vs uninitialized models."""

    # Calculate the bootstrapped p-values for the init and uninit models
    # first for the initialized models - dcppA-hindcast
    # Only do 100 bootstraps for now
    pfield_bs_dcpp = fnc.calculate_spatial_correlations_bootstrap(observed_data, init_model_data, init_models, variable, n_bootstraps=100)

    # Then for the uninitialized models - historical
    # Only do 100 bootstraps for now
    pfield_bs_hist = fnc.calculate_spatial_correlations_bootstrap(observed_data, uninit_model_data, uninit_models, variable, n_bootstraps=100)
    
    # First source the data for this function
    # Using the calculate_spatial_correlations function
    # First for the init models (dcppA-hindcast)
    rfield_init, pfield_init, obs_lons_converted_init, lons_converted_init, \
        _, ensemble_mean_init, ensemble_members_count_init = fnc.calculate_spatial_correlations(observed_data,
                                                                                                    init_model_data, init_models,
                                                                                                        variable)
    
    # Then for the uninit models (historical)
    rfield_uninit, pfield_uninit, obs_lons_converted_uninit, lons_converted_uninit, \
        observed_data_array, ensemble_mean_uninit, \
            ensemble_members_count_uninit = fnc.calculate_spatial_correlations(observed_data,
                                                                                uninit_model_data, uninit_models,
                                                                                    variable)

    # Calculate the difference between the init and uninit rfields
    rfield_diff = rfield_init - rfield_uninit                                                                                

    # Print the types of the rfield_init and rfield_uninit
    # print("rfield_init type", type(rfield_init))
    # print("rfield_uninit type", type(rfield_uninit))

    # # print the types of the pfield_init and pfield_uninit
    # print("pfield_init type", type(pfield_init))
    # print("pfield_uninit type", type(pfield_uninit))

    # # print the shapes of the rfield_init and rfield_uninit
    # print("rfield_init shape", rfield_init.shape)
    # print("rfield_uninit shape", rfield_uninit.shape)

    # # print the values of the rfield_init and rfield_uninit
    # print("rfield_init values", rfield_init)
    # print("rfield_uninit values", rfield_uninit)

    # Set the font size for the plots
    plt.rcParams.update({'font.size': 12})

    # set up the proj
    proj = ccrs.PlateCarree()

    # Extract the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid['lon1'], azores_grid['lon2']
    azores_lat1, azores_lat2 = azores_grid['lat1'], azores_grid['lat2']

    # Extract the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid['lon1'], iceland_grid['lon2']
    iceland_lat1, iceland_lat2 = iceland_grid['lat1'], iceland_grid['lat2']

    # Extract the lats and lons for the northern UK index box
    uk_n_lon1, uk_n_lon2 = uk_n_box['lon1'], uk_n_box['lon2']
    uk_n_lat1, uk_n_lat2 = uk_n_box['lat1'], uk_n_box['lat2']

    # Extract the lats and lons for the southern UK index box
    uk_s_lon1, uk_s_lon2 = uk_s_box['lon1'], uk_s_box['lon2']
    uk_s_lat1, uk_s_lat2 = uk_s_box['lat1'], uk_s_box['lat2']

    # subtract 180 from all of the azores and iceland lons
    azores_lon1, azores_lon2 = azores_lon1 - 180, azores_lon2 - 180
    iceland_lon1, iceland_lon2 = iceland_lon1 - 180, iceland_lon2 - 180

    # subtract 180 from all of the uk lons
    uk_n_lon1, uk_n_lon2 = uk_n_lon1 - 180, uk_n_lon2 - 180
    uk_s_lon1, uk_s_lon2 = uk_s_lon1 - 180, uk_s_lon2 - 180

    # set up the converted lons
    # Set the lons as the init lons
    # these are the same for both init and uninit
    lons_converted = lons_converted_init
    lons_converted = lons_converted - 180

    # Set up the lats and lons
    # if the region is global
    if region == 'global':
        lats = observed_data.lat
        lons = lons_converted
    # if the region is not global
    elif region == 'north-atlantic':
        lats = observed_data.lat
        lons = lons_converted
    else:
        print("Error: region not found")
        sys.exit()

    # Set up the significance threshold
    # if p_sig is 0.05, then sig_threshold is 95%
    sig_threshold = int((1 - p_sig) * 100)

    # if observed_data is not None:
    # Extract the first and last years
    if observed_data is not None:
        first_year = observed_data.time.dt.year.values[0]
        last_year = observed_data.time.dt.year.values[-1]
    else:
        first_year = None
        last_year = None

    # If ensemble_members_count is not None
    if ensemble_members_count_init is not None:
        total_no_members_init = sum(ensemble_members_count_init.values())
    else:
        total_no_members = None

    # If ensemble_members_count is not None
    if ensemble_members_count_uninit is not None:
        total_no_members_uninit = sum(ensemble_members_count_uninit.values())
    else:
        total_no_members = None

    # Set up pfield diff as an array of Nan's
    # with the same shape as rfield_diff
    pfield_diff = np.empty_like(rfield_diff)
    # Fill pfield_diff with nan's
    pfield_diff.fill(np.nan)

    # print the shape of pfield_diff
    print("pfield_diff shape", pfield_diff.shape)
    # print the shape of rfield_diff
    print("rfield_diff shape", rfield_diff.shape)

    # print the values of pfield_diff
    print("pfield_diff values", pfield_diff)

    # create a list of the rfield_init and rfield_uninit to be plotted
    rfield_list = [rfield_init, rfield_uninit, rfield_init, rfield_uninit, rfield_diff]

    # same for the pfield_init and pfield_uninit
    pfield_list = [pfield_init, pfield_uninit, pfield_bs_dcpp, pfield_bs_hist, pfield_diff]


    # Set up the figure as two subplots (1 row, 2 columns)
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 12), subplot_kw={'projection': proj}, gridspec_kw={'wspace': 0.1})
    # Remove the last subplot
    axs[-1, -1].remove()

    # flatten the axs array
    axs = axs.flatten()

    # Create a list to store the contourf objects
    cf_list = []

    for i, rfield in enumerate(rfield_list):

        # print the rfield
        rfield = rfield_list[i]

        # print the pfield
        pfield = pfield_list[i]

        # # Print the r and p fields
        # print("plotting rfield", rfield)
        # print("plotting pfield", pfield)

        # set up the axes
        ax = axs[i]

        # Add coastlines
        ax.coastlines()
        
        # Add green lines outlining the Azores and Iceland grids
        ax.plot([azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1], [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1], color='green', linewidth=2, transform=proj)
        ax.plot([iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1], [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1], color='green', linewidth=2, transform=proj)

        # add filled contours
        # Contour levels
        clevs = np.arange(-1.8, 1.8, 0.1)
        # Contour levels for p-values
        clevs_p = np.arange(0, 1.1, 0.1)
        # Plot the filled contours
        cf = ax.contourf(lons, lats, rfield, clevs, cmap='RdBu_r', transform=proj)

        # replace values in pfield that are greater than 0.05 with nan
        pfield[pfield > p_sig] = np.nan

        # Add stippling where rfield is significantly different from zero
        ax.contourf(lons, lats, pfield, hatches=['....'], alpha=0, transform=proj)

        # if i == 0:
        # add textbox
        if i == 0:
            # Add a textbox for the first subplot
            # in the top left
            # with 'dcppA' in it
            ax.text(0.05, 0.95, 'dcppA', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))

            # Add another textbox for the first subplot
            # in the bottom right
            # for the number of ensemble members
            ax.text(0.95, 0.05, f"N = {total_no_members_init}", transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.5))
        elif i == 1:
            # Add a textbox for the second subplot
            # in the top left
            # with 'historical' in it
            ax.text(0.05, 0.95, 'historical', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))

            # Add another textbox for the second subplot
            # in the bottom right
            # for the number of ensemble members
            ax.text(0.95, 0.05, f"N = {total_no_members_uninit}", transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.5))
        elif i == 2:
            # Add a textbox for the third subplot
            # in the top left
            # with 'dcppA - historical' in it
            ax.text(0.05, 0.95, 'dcppA (bs pvalues)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))
        elif i == 3:
            # Add a textbox for the third subplot
            # in the top left
            # with 'dcppA - historical' in it
            ax.text(0.05, 0.95, 'historical (bs pvalues)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))
        elif i == 4:
            # Add a textbox for the third subplot
            # in the top left
            # with 'dcppA - historical' in it
            ax.text(0.05, 0.95, 'init - uninit', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))
        else:
            print("Error: subplot not found")

        # Append the contourf object to the list
        cf_list.append(cf)

    # Add colorbar
    cbar = plt.colorbar(cf_list[0], orientation='horizontal', pad=0.05, aspect=50, ax=fig.axes, shrink=0.8)
    cbar.set_label('correlation coefficients')

    # extract the model name from the list
    # given as ['model']
    # we only want the model name
    # if the length of the list is 1
    # then the model name is the first element
    # set models as init_models
    model = init_models

    # if the length of the list is 1
    if len(model) == 1:
        model = model[0]
    elif len(model) > 1:
        model = "multi-model mean"
    else :
        print("Error: model name not found")
        sys.exit()

    # Set up the title
    title = f"{model} {variable} {region} {season} {forecast_range} {first_year}-{last_year} correlation coefficients, p < {p_sig} ({sig_threshold}%)"

    # Add the sup title
    fig.suptitle(title, fontsize=12)

    # set up the figure name
    fig_name = f"{model}_{variable}_{region}_{season}_{forecast_range}_{total_no_members_init}_{total_no_members_uninit}_{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    # Set up the figure path
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # Show the figure
    plt.show()

# SUbplots function
# Function for plotting the results for all of the models as 12 subplots
# FIXME: get this working for model differences
def plot_correlations_subplots(models, obs, variable_data, variable, region, season, forecast_range, plots_dir, azores_grid, iceland_grid, uk_n_box, uk_s_box, p_sig = 0.05, experiment=None, observed_data=None):
    """Plot the spatial correlation coefficients and p-values for all models.

    This function plots the spatial correlation coefficients and p-values
    for all models in the dictionaries.models list for a given variable,
    region, season and forecast range.

    Parameters
    ----------
    models : List
        List of models.
    obs : str
        Observed dataset.
    variable_data : dict
        Variable data for each model.
    region : str
        Region.
    season : str
        Season.
    forecast_range : str
        Forecast range.
    plots_dir : str
        Path to the directory where the plots will be saved.
    azores_grid : array
        Array of longitudes and latitudes for the Azores region.
    iceland_grid : array
        Array of longitudes and latitudes for the Iceland region.
    uk_n_box : array
        Array of longitudes and latitudes for the northern UK index box.
    uk_s_box : array
        Array of longitudes and latitudes for the southern UK index box.
    p_sig : float, optional
        Significance threshold. The default is 0.05.
    experiment : str, optional
        Experiment. The default is None.
    observed_data : xarray.Dataset, optional
        Observed data. The default is None.
    """

    # Set the font size for the plots
    plt.rcParams.update({'font.size': 12})

    # Set the projection
    proj = ccrs.PlateCarree()

    # Set up the first and last years
    if observed_data is not None:
        first_year = observed_data.time.dt.year.values[0]
        last_year = observed_data.time.dt.year.values[-1]
    else:
        first_year = None
        last_year = None
    
    # Set up the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid['lon1'], azores_grid['lon2']
    azores_lat1, azores_lat2 = azores_grid['lat1'], azores_grid['lat2']

    # Set up the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid['lon1'], iceland_grid['lon2']
    iceland_lat1, iceland_lat2 = iceland_grid['lat1'], iceland_grid['lat2']
    
    # Set up the lats and lons for the northern UK index box
    uk_n_lon1, uk_n_lon2 = uk_n_box['lon1'], uk_n_box['lon2']
    uk_n_lat1, uk_n_lat2 = uk_n_box['lat1'], uk_n_box['lat2']

    # Set up the lats and lons for the southern UK index box
    uk_s_lon1, uk_s_lon2 = uk_s_box['lon1'], uk_s_box['lon2']
    uk_s_lat1, uk_s_lat2 = uk_s_box['lat1'], uk_s_box['lat2']

    # subtract 180 from all of the azores and iceland lons
    azores_lon1, azores_lon2 = azores_lon1 - 180, azores_lon2 - 180
    iceland_lon1, iceland_lon2 = iceland_lon1 - 180, iceland_lon2 - 180

    # subtract 180 from all of the uk lons
    uk_n_lon1, uk_n_lon2 = uk_n_lon1 - 180, uk_n_lon2 - 180
    uk_s_lon1, uk_s_lon2 = uk_s_lon1 - 180, uk_s_lon2 - 180

    # Count the number of models available
    nmodels = len(models)

    # Set the figure size and subplot parameters
    if nmodels == 8:
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), subplot_kw={'projection': proj}, gridspec_kw={'wspace': 0.1})
        # Remove the last subplot
        axs[-1, -1].remove()
        # Set up where to plot the title
        title_index = 1
    elif nmodels == 10:
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 16), subplot_kw={'projection': proj}, gridspec_kw={'wspace': 0.1})
        # remove the last subplot
        axs[-1, -1].remove()
        # remove the second last subplot
        axs[-1, -2].remove()
        # Set up where to plot the title
        title_index = 1
    elif nmodels == 11:
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 16), subplot_kw={'projection': proj}, gridspec_kw={'wspace': 0.1})
        axs[-1, -1].remove()
        # Set up where to plot the title
        title_index = 1
    elif nmodels == 12:
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 16), subplot_kw={'projection': proj}, gridspec_kw={'wspace': 0.1})
        # Set up where to plot the title
        title_index = 1
    else:
        raise ValueError(f"Invalid number of models: {nmodels}")
    
    # Set up the significance threshold
    # e.g. 0.05 for 95% significance
    sig_threshold = int((1 - p_sig) * 100)
    
    # Flatten the axs array
    axs = axs.flatten()

    # Create a list to store the contourf objects
    cf_list = []
    
    # Loop over the models
    for i, model in enumerate(models):
        
        # #print the model name
        #print("Processing model:", model)
    
        # Convert the model to a single index list
        model = [model]
    
        # Calculate the spatial correlations for the model
        rfield, pfield, obs_lons_converted, lons_converted, observed_data, ensemble_mean, ensemble_members_count = calculate_spatial_correlations(obs, variable_data, model, variable)

        # Set up the converted lons
        lons_converted = lons_converted - 180

        # Set up the lats and lons
        # if the region is global
        if region == 'global':
            lats = obs.lat
            lons = lons_converted
        # if the region is not global
        elif region == 'north-atlantic':
            lats = obs.lat
            lons = lons_converted
        else:
            #print("Error: region not found")
            sys.exit()

        # Set up the axes
        ax = axs[i]

        # Add coastlines
        ax.coastlines()
    
        # Add gridlines with labels for the latitude and longitude
        # gl = ax.gridlines(crs=proj, draw_labels=False, linewidth=2, color='gray', alpha=0.5, linestyle='--')
        # gl.top_labels = False
        # gl.right_labels = False
        # gl.xlabel_style = {'size': 12}
        # gl.ylabel_style = {'size': 12}
    
        # Add green lines outlining the Azores and Iceland grids
        ax.plot([azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1], [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1], color='green', linewidth=2, transform=proj)
        ax.plot([iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1], [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1], color='green', linewidth=2, transform=proj)

        # Add green lines outlining the northern and southern UK index boxes
        ax.plot([uk_n_lon1, uk_n_lon2, uk_n_lon2, uk_n_lon1, uk_n_lon1], [uk_n_lat1, uk_n_lat1, uk_n_lat2, uk_n_lat2, uk_n_lat1], color='green', linewidth=2, transform=proj)
        ax.plot([uk_s_lon1, uk_s_lon2, uk_s_lon2, uk_s_lon1, uk_s_lon1], [uk_s_lat1, uk_s_lat1, uk_s_lat2, uk_s_lat2, uk_s_lat1], color='green', linewidth=2, transform=proj)
    
        # Add filled contours
        # Contour levels
        clevs = np.arange(-1.8, 1.8, 0.1)
        # Contour levels for p-values
        clevs_p = np.arange(0, 1.1, 0.1)
        # Plot the filled contours
        cf = ax.contourf(lons, lats, rfield, clevs, cmap='RdBu_r', transform=proj)

        # replace values in pfield that are greater than 0.01 with nan
        pfield[pfield > p_sig] = np.nan
    
        # Add stippling where rfield is significantly different from zero
        ax.contourf(lons, lats, pfield, hatches=['....'], alpha=0, transform=proj)
    
        # Add title
        # ax.set_title(f"{model} {variable} {region} {season} {forecast_range} Correlation Coefficients")
    
        # extract the model name from the list
        if len(model) == 1:
            model = model[0]
        elif len(model) > 1:
            model = "all_models"
        else :
            #print("Error: model name not found")
            sys.exit()
    
        # Add textbox with model name
        ax.text(0.05, 0.95, model, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.5))

        # Add a textbook with the number of ensemble members
        total_no_members = sum(ensemble_members_count.values())
        # Include this textbox in the bottom right corner
        ax.text(0.95, 0.05, f"N = {total_no_members}", transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.5))
    
        # Add the contourf object to the list
        cf_list.append(cf)

        # If this is the centre subplot on the first row, set the title for the figure
        if i == title_index:
            # Add title
            ax.set_title(f"{variable} {region} {season} years {forecast_range} {experiment} {first_year}-{last_year} correlation coefficients, p < {p_sig} ({sig_threshold}%)", fontsize=12)
    
    # Create a single colorbar for all of the subplots
    cbar = plt.colorbar(cf_list[0], orientation='horizontal', pad=0.05, aspect=50, ax=fig.axes, shrink=0.8)
    cbar.set_label('Correlation Coefficient')
    
    # Specify a tight layout
    # plt.tight_layout()

    # if experiment is not None:
    if experiment is not None:
        fig_name = f"{variable}_{region}_{season}_{forecast_range}_{experiment}_{first_year}-{last_year}_sig-{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    else:
        # set up the path for saving the figure
        fig_name = f"{variable}_{region}_{season}_{forecast_range}_{first_year}_{last_year}_sig-{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    # Set up the figure path
    fig_path = os.path.join(plots_dir, fig_name)

    # # Adjust the vertical spacing between the plots
    # plt.subplots_adjust(hspace=0.1)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    # Show the figure
    plt.show()


def plot_seasonal_correlations_diff(observations_path, historical_models, dcpp_models, variable, region, region_grid,
                                    forecast_range, seasons_list_obs, seasons_list_mod, plots_dir,
                                    obs_var_name, azores_grid, iceland_grid, p_sig=0.05, experiment='differences', n_bootstraps=100):
    """
    Plot the spatial correlation coefficients differences and p-values for all models for each season.

    This function takes a list of models, the path to the observations file, the initialized and uninitialized model
    data, the names of the initialized and uninitialized models, the variable, region, region grid, forecast range,
    lists of seasons for the observations and models, the directory where the plots will be saved, the name of the
    observed variable, the Azores grid, the Iceland grid, the p-value threshold, the experiment name (optional), and
    the number of bootstraps to use for the confidence intervals (optional). It then calculates the correlation
    coefficients and p-values for each model and season and plots the differences between the initialized and
    uninitialized simulations as four subplots, one for each season: DJFM, MAM, JJA, SON.

    Parameters
    ----------
    observations_path : str
        The path to the observations file.
    historical_models : list of str
        The names of the historical models.
    dcpp_models : list of str   
        The names of the DCPP models.
    variable : str
        The variable to plot.
    region : str
        The region to plot.
    region_grid : numpy.ndarray
        The longitudes and latitudes for the region.
    forecast_range : str
        The forecast range to plot.
    seasons_list_obs : list of str
        The seasons to use for the observations.
    seasons_list_mod : list of str
        The seasons to use for the models.
    plots_dir : str
        The directory where the plots will be saved.
    obs_var_name : str
        The name of the observed variable.
    azores_grid : tuple of float
        The longitude and latitude of the Azores.
    iceland_grid : tuple of float
        The longitude and latitude of Iceland.
    p_sig : float, optional
        The p-value threshold for significance. Default is 0.05.
    experiment : str, optional
        The name of the experiment. Default is None.
    n_bootstraps : int, optional
        The number of bootstraps to use for the confidence intervals. Default is 100.

    Returns
    -------
    None
    """

    # Create an empty list to store the processed observations
    # for each season
    obs_list = []

    # Create empty lists to store the rfield and pfield
    # for each season
    rfield_diff_list = []
    pfield_list = []

    # Create lists to store the obs_lons_converted and lons_converted
    # for each season
    obs_lons_converted_list = []
    lons_converted_list = []

    # Create the list of labels to add to the subplot
    ax_labels = ['A', 'B', 'C', 'D']

    # Loop over the seasons
    for i, season in enumerate(seasons_list_obs):

        # Print the seasons being processed
        print("Processing season:", season)
        print("Processing season:", seasons_list_mod[i])

        # Process the observations for this season
        obs = fnc.process_observations(variable, region, region_grid, forecast_range,
                                        season, observations_path, obs_var_name)
        # Append the processed observations to the list
        obs_list.append(obs)

        # Load and process the uninitialized model data (historical) for this season
        historical_data = fnc.load_processed_historical_data(dic.base_dir_historical, historical_models,
                                                            variable, region, forecast_range,
                                                            seasons_list_mod[i])
        # Process the uninitialized model data (historical) for this season
        historical_data, _ = fnc.extract_historical_data(historical_data, variable)

        # Load and process the initialized model data (dcpp) for this season
        dcpp_data = fnc.load_data(dic.dcpp_base_dir, dcpp_models,
                                    variable, region, forecast_range,
                                    seasons_list_mod[i])
        # Process the initialized model data (dcpp) for this season
        dcpp_data, _ = fnc.process_data(dcpp_data, variable)


        # Get the rfield_diff
        # using the calculate_spatial_correlations_diff function
        rfield_diff, sign_regions, obs_lons_converted, lons_converted, observed_data, \
        dcpp_ensemble_mean, historical_ensemble_mean, dcpp_ensemble_members_count, \
        historical_ensemble_members_count = fnc.calculate_spatial_correlations_diff(obs, dcpp_data, historical_data,
                                                                                    dcpp_models, historical_models,
                                                                                    variable)
        # Append the rfield_diff to the list
        rfield_diff_list.append(rfield_diff)

        # Get the bootstrapped pfield
        # using the calculate_spatial_correlations_bootstrapp_diff function
        pfield_diff_bs = fnc.calculate_spatial_correlations_bootstrap_diff(obs, dcpp_data, historical_data,
                                                                            dcpp_models, historical_models,
                                                                            variable, n_bootstraps)
        # Append the pfield_diff_bs to the list
        pfield_list.append(pfield_diff_bs)

        # Append the obs_lons_converted and lons_converted to the lists
        obs_lons_converted_list.append(obs_lons_converted)
        lons_converted_list.append(lons_converted)

    # Set the font size for the plots
    plt.rcParams.update({'font.size': 12})

    # Set the projection
    proj = ccrs.PlateCarree()

    # Set up the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid['lon1'], azores_grid['lon2']
    azores_lat1, azores_lat2 = azores_grid['lat1'], azores_grid['lat2']

    # Set up the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid['lon1'], iceland_grid['lon2']
    iceland_lat1, iceland_lat2 = iceland_grid['lat1'], iceland_grid['lat2']

    # subtract 180 from all of the azores and iceland lons
    azores_lon1, azores_lon2 = azores_lon1 - 180, azores_lon2 - 180
    iceland_lon1, iceland_lon2 = iceland_lon1 - 180, iceland_lon2 - 180

    # Set up the fgure size and subplot parameters
    # Set up the fgure size and subplot parameters
    # for a 2x2 grid of subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), subplot_kw={'projection': proj}, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

    # Set up the title for the figure
    title = f"{variable} {region} {forecast_range} {experiment} correlation coefficients, p < {p_sig} ({int((1 - p_sig) * 100)}%)"

    # Set up the supertitle for the figure
    fig.suptitle(title, fontsize=12, y=0.90)

    # Set up the significance thresholdf
    # e.g. 0.05 for 95% significance
    sig_threshold = int((1 - p_sig) * 100)

    # Flatten the axs array
    axs = axs.flatten()

    # Create a list to store the contourf objects
    cf_list = []

    # Loop over the seasons
    for i, season in enumerate(seasons_list_obs):

        # Print the season being plotted
        print("Plotting season:", season)

        # Extract the obs
        obs = obs_list[i]

        # Extract the rfield_diff
        rfield_diff = rfield_diff_list[i]

        # Extract the pfield
        pfield_bs = pfield_list[i]

        # Extract the obs_lons_converted and lons_converted
        obs_lons_converted = obs_lons_converted_list[i]
        lons_converted = lons_converted_list[i]

        # Set up the converted lons
        lons_converted = lons_converted - 180

        # Set up the lats and lons
        # if the region is global
        lats = obs.lat
        lons = lons_converted

        # Set up the axes
        ax = axs[i]

        # Add coastlines
        ax.coastlines()

        # Add greenlines outlining the Azores and Iceland grids
        ax.plot([azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1], [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1], color='green', linewidth=2, transform=proj)
        ax.plot([iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1], [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1], color='green', linewidth=2, transform=proj)

        # Add filled contours
        # Contour levels
        clevs = np.arange(-1.8, 1.8, 0.1)
        # Contour levels for p-values
        clevs_p = np.arange(0, 1.1, 0.1)
        # Plot the filled contours
        cf = ax.contourf(lons, lats, rfield_diff, clevs, cmap='RdBu_r', transform=proj)

        # replace values in pfield that are greater than 0.05 with nan
        pfield_bs[pfield_bs > p_sig] = np.nan

        # Add stippling where rfield is significantly different from zero
        ax.contourf(lons, lats, pfield_bs, hatches=['....'], alpha=0, transform=proj)

        # Add a textbox with the season name
        ax.text(0.05, 0.95, season, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.5))

        # Add a textbox with the label
        fig_letter = ax_labels[i]
        ax.text(0.95, 0.05, fig_letter, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.5))

        # Append the contourf object to the list
        cf_list.append(cf)

    # Add colorbar
    cbar = plt.colorbar(cf_list[0], orientation='horizontal', pad=0.05, aspect=50, ax=fig.axes, shrink=0.8)
    cbar.set_label('correlation coefficient differences (init - uninit)')

    # Set up the figure name
    fig_name = f"{variable}_{region}_{forecast_range}_{experiment}_correlation_coefficients_diff_subplots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # Show the figure
    plt.show()

