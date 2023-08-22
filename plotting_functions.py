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
    clevs = np.arange(-1, 1.1, 0.1)
    # Contour levels for p-values
    clevs_p = np.arange(0, 1.1, 0.1)
    # Plot the filled contours
    cf = plt.contourf(lons, lats, rfield, clevs, cmap='RdBu_r', transform=ccrs.PlateCarree())

    # replace values in pfield that are greater than 0.05 with nan
    pfield[pfield > p_sig] = np.nan

    # print the pfield
    # print("pfield mod", pfield)

    # Add stippling where rfield is significantly different from zero
    # plt.contourf(lons, lats, pfield, hatches=['....'], alpha=0, transform=ccrs.PlateCarree())

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


# Define a function to plot subplots of the correlation coefficients and p-values for the init vs uninitialized models
def plot_correlations_init_vs_uninit(observed_data, init_model_data, uninit_model_data, init_models, uninit_models, variable, region, season, forecast_range, plots_dir, azores_grid, iceland_grid, uk_n_box, uk_s_box, p_sig=0.05):
    """Plot the correlation coefficients and p-values for the init vs uninitialized models."""

    # First source the data for this function
    # Using the calculate_spatial_correlations function
    # First for the init models (dcppA-hindcast)
    rfield_init, pfield_init, obs_lons_converted_init, lons_converted_init, \
        _, ensemble_mean_init, ensemble_members_count_init = fnc.calculate_spatial_correlations(observed_data,
                                                                                                    init_model_data, init_models,
                                                                                                        variable)
    
    # Then for the uninit models (historical)
    rfield_uninit, pfield_uninit, obs_lons_converted_uninit, lons_converted_uninit, \
        observed_data, ensemble_mean_uninit, \
            ensemble_members_count_uninit = fnc.calculate_spatial_correlations(observed_data,
                                                                                uninit_model_data, uninit_models,
                                                                                    variable)
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

    # Set up the figure as two subplots (1 row, 2 columns)
    fig, axs = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set up the first subplot
    ax1 = axs[0]
    ax2 = axs[1]

    # Set up the projection
    ax1 = plt.axes(projection=ccrs.PlateCarree())
    ax2 = plt.axes(projection=ccrs.PlateCarree())

    # Add coastlines
    ax1.coastlines()
    ax2.coastlines()

    # Add green lines outlining the Azores and Iceland grids
    ax1.plot([azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1], [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())
    ax1.plot([iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1], [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())

    # Same for the second subplot
    ax2.plot([azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1], [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())
    ax2.plot([iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1], [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())

    # Set up the contour levels
    # Contour levels
    clevs = np.arange(-1, 1.1, 0.1)
    # Contour levels for p-values
    clevs_p = np.arange(0, 1.1, 0.1)

    # Plot the filled contours on the first subplot
    cf1 = ax1.contourf(lons, lats, rfield_init, clevs, cmap='RdBu_r', transform=ccrs.PlateCarree())
    # Plot the filled contours on the second subplot
    cf2 = ax2.contourf(lons, lats, rfield_uninit, clevs, cmap='RdBu_r', transform=ccrs.PlateCarree())

    # Replace values in pfield that are greater than 0.05 with nan
    pfield_init[pfield_init > p_sig] = np.nan
    pfield_uninit[pfield_uninit > p_sig] = np.nan

    # Add stippling where rfield is significantly different from zero
    ax1.contourf(lons, lats, pfield_init, hatches=['....'], alpha=0, transform=ccrs.PlateCarree())
    ax2.contourf(lons, lats, pfield_uninit, hatches=['....'], alpha=0, transform=ccrs.PlateCarree())

    # Add a constant colorbar for both subplots
    cbar = plt.colorbar(cf1, orientation='horizontal', pad=0.05, aspect=50, ax=axs[:], shrink=0.8)
    cbar.set_label('correlation coefficients')

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
    if ensemble_members_count_init is not None:
        total_no_members_init = sum(ensemble_members_count_init.values())
    else:
        total_no_members = None

    # If ensemble_members_count is not None
    if ensemble_members_count_uninit is not None:
        total_no_members_uninit = sum(ensemble_members_count_uninit.values())
    else:
        total_no_members = None

    # Add a textbox for the first subplot
    # in the top right
    # with 'dcppA' in it
    ax1.text(0.95, 0.95, 'dcppA', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', ha='right', bbox=dict(facecolor='white', alpha=0.5))

