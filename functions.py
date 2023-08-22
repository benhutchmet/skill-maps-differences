# Functions for creating the skill map differences

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

# Define a new function to load the processed historical data
# This function will take as arguments: the base directory 
# where the data are stored, the models, the variable name, the region name, the forecast range and the season.
def load_processed_historical_data(base_dir, models, variable, region, forecast_range, season):
    """
    Load the processed historical data from its base directory into a dictionary of datasets.

    Arguments:
        base_dir -- the base directory where the data are stored
        models -- the models
        variable -- the variable name
        region -- the region name
        forecast_range -- the forecast range

    Returns:
        historical_data -- a dictionary of datasets grouped by model and member.
    """

    # Initialize the dictionary to store the data
    historical_data = {}

    # Loop over the models
    for model in models:
            
        # Print the model name
        print("processing model: ", model)

        # Create an empty list to store the datasets
        # within the dictionary
        historical_data[model] = []

        # Set up the file path for the model
        # First the directory path
        # base_dir = "/home/users/benhutch/skill-maps-processed-data/historical"
        files_path = base_dir + '/' + variable + '/' + model + '/' + region + '/' + 'years_' + forecast_range + '/' + season + '/' + 'outputs' + '/' + 'processed' + '/' + '*.nc'

        # Print the files path
        print("files_path: ", files_path)

        # Find the files which match the path
        files = glob.glob(files_path)

        # If the list of files is empty, then print a warning and exit
        if len(files) == 0:
            print("Warning, no files found for model: ", model)
            return None

        # Loop over the files
        for file in files:
            # Open the dataset using xarray
            data = xr.open_dataset(file, chunks={'time': 100, 'lat': 45, 'lon': 45})

            # Extract the variant_label
            variant_label = data.attrs['variant_label']

            # Print the variant_label
            print("loading variant_label: ", variant_label)

            # Add the data to the dictionary
            # Using the variant_label as the key
            historical_data[model].append(data)

    # Return the historical data dictionary
    return historical_data

# Set up a function which which given the dataset and variable
# will extract the data for the given variable and the time dimension
def process_historical_members(model_member, variable):
    """
    For a given model member, extract the data for the given variable and the time dimension.
    """
    
    # Print the variable name
    print("processing variable: ", variable)

    # Check that the variable name is valid
    if variable not in dic.variables:
        print("Error, variable name not valid")
        return None

    try:
        # Extract the data for the given variable
        variable_data = model_member[variable]
    except Exception as error:
        print(error)
        print("Error, failed to extract data for variable: ", variable)
        return None
    
    # Extract the time dimension
    try:
        # Extract the time dimension
        historical_time = model_member["time"].values

        # Change these to a year format
        historical_time = historical_time.astype('datetime64[Y]').astype(int) + 1970
    except Exception as error:
        print(error)
        print("Error, failed to extract time dimension")
        return None
    
    # If either the variable data or the time dimension are None
    # then return None
    if variable_data is None or historical_time is None:
        print("Error, variable data or time dimension is None")
        sys.exit(1)

    return variable_data, historical_time

# Now write the outer function which will call this inner function
def extract_historical_data(historical_data, variable):
    """
    Outer function to process the historical data by extracting the data 
    for the given variable and the time dimension.
    """

    # Create empty dictionaries to store the data
    variable_data_by_model = {}
    historical_time_by_model = {}

    # Loop over the models
    for model in historical_data:
        # Print the model name
        print("processing model: ", model)

        # Create empty lists to store the data
        variable_data_by_model[model] = []
        historical_time_by_model[model] = []

        # Loop over the members
        for member in historical_data[model]:
            # Print the member name
            print("processing member: ", member)

            # print the type of the historical data
            print("type of historical_data: ", type(member))

            # print the dimensions of the historical data
            print("dimensions of historical_data: ", member.dims)

            # Format as a try except block    
            try:
                # Process the historical data
                variable_data, historical_time = process_historical_members(member, variable)

                # Append the data to the list
                variable_data_by_model[model].append(variable_data)
                historical_time_by_model[model].append(historical_time)
            except Exception as error:
                print(error)
                print("Error, failed to process historical data using process_historical_members in outer function")
                return None
            
    # Return the data
    return variable_data_by_model, historical_time_by_model

# Function to constrain the years of the historical data
# Define a function to constrain the years to the years that are in all of the model members
def constrain_years_processed_hist(model_data, models):
    """
    Constrains the years to the years that are in all of the models.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.

    Returns:
    constrained_data (dict): The model data with years constrained to the years that are in all of the models.
    """
    # Initialize a list to store the years for each model
    years_list = []

    # Print the models being proces
    # print("models:", models)
    
    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            
            # Extract the years
            years = member.time.dt.year.values

            # If years is less than 48, then don't use this member
            if len(years) < 48:
                print("years less than 48")
                print("not including this member in common years for this model: ", model)
                continue

            # Append the years to the list of years
            years_list.append(years)

    # Find the years that are in all of the models
    common_years = list(set(years_list[0]).intersection(*years_list))

    # Print the common years for debugging
    # print("Common years:", common_years)
    # print("Common years type:", type(common_years))
    # print("Common years shape:", np.shape(common_years))

    # Initialize a dictionary to store the constrained data
    constrained_data = {}

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Extract the years
            years = member.time.dt.year.values

            # Print the years extracted from the model
            # print('model years', years)
            # print('model years shape', np.shape(years))

            # If years is less than 48, then don't use this member
            if len(years) < 48:
                print("years less than 48")
                print("not using this member for this model: ", model)
                continue
            
            # Find the years that are in both the model data and the common years
            years_in_both = np.intersect1d(years, common_years)

            # print("years in both shape", np.shape(years_in_both))
            # print("years in both", years_in_both)
            
            # Select only those years from the model data
            member = member.sel(time=member.time.dt.year.isin(years_in_both))

            # Add the member to the constrained data dictionary
            if model not in constrained_data:
                constrained_data[model] = []
            constrained_data[model].append(member)

    # # Print the constrained data for debugging
    # print("Constrained data:", constrained_data)

    return constrained_data

# Function to remove years with Nans
# checking for Nans in observed data
def remove_years_with_nans(observed_data, ensemble_mean, variable):
    """
    Removes years from the observed data that contain NaN values.

    Args:
        observed_data (xarray.Dataset): The observed data.
        ensemble_mean (xarray.Dataset): The ensemble mean (model data).
        variable (str): the variable name.

    Returns:
        xarray.Dataset: The observed data with years containing NaN values removed.
    """

    # # Set the obs_var_name == variable
    obs_var_name = variable

    # Set up the obs_var_name
    if obs_var_name == "psl":
        obs_var_name = "msl"
    elif obs_var_name == "tas":
        obs_var_name = "t2m"
    elif obs_var_name == "sfcWind":
        obs_var_name = "si10"
    elif obs_var_name == "rsds":
        obs_var_name = "ssrd"
    elif obs_var_name == "tos":
        obs_var_name = "sst"
    else:
        print("Invalid variable name")
        sys.exit()

    print("var name for obs", obs_var_name)
    
    for year in observed_data.time.dt.year.values[::-1]:
        # Extract the data for the year
        data = observed_data.sel(time=f"{year}")

        
        # If there are any Nan values in the data
        if np.isnan(data.values).any():
            # Print the year
            # print(year)

            # Select the year from the observed data
            observed_data = observed_data.sel(time=observed_data.time.dt.year != year)

            # for the model data
            ensemble_mean = ensemble_mean.sel(time=ensemble_mean.time.dt.year != year)

        # if there are no Nan values in the data for a year
        # then print the year
        # and "no nan for this year"
        # and continue the script
        else:
            # print(year, "no nan for this year")

            # exit the loop
            break

    return observed_data, ensemble_mean

# Function for calculating the spatial correlations
def calculate_correlations(observed_data, model_data, obs_lat, obs_lon):
    """
    Calculates the spatial correlations between the observed and model data.

    Parameters:
    observed_data (numpy.ndarray): The processed observed data.
    model_data (numpy.ndarray): The processed model data.
    obs_lat (numpy.ndarray): The latitude values of the observed data.
    obs_lon (numpy.ndarray): The longitude values of the observed data.

    Returns:
    rfield (xarray.core.dataarray.DataArray): The spatial correlations between the observed and model data.
    pfield (xarray.core.dataarray.DataArray): The p-values for the spatial correlations between the observed and model data.
    """
    try:
        # Initialize empty arrays for the spatial correlations and p-values
        rfield = np.empty([len(obs_lat), len(obs_lon)])
        pfield = np.empty([len(obs_lat), len(obs_lon)])

        # Print the dimensions of the observed and model data
        # print("observed data shape", np.shape(observed_data))
        # print("model data shape", np.shape(model_data))

        # Loop over the latitudes and longitudes
        for y in range(len(obs_lat)):
            for x in range(len(obs_lon)):
                # set up the obs and model data
                obs = observed_data[:, y, x]
                mod = model_data[:, y, x]

                # print the obs and model data
                # print("observed data", obs)
                # print("model data", mod)

                # Calculate the correlation coefficient and p-value
                r, p = stats.pearsonr(obs, mod)

                # If the correlation coefficient is negative, set the p-value to NaN
                if r < 0:
                    p = np.nan

                # Append the correlation coefficient and p-value to the arrays
                rfield[y, x], pfield[y, x] = r, p

        # Print the range of the correlation coefficients and p-values
        # to 3 decimal places
        print(f"Correlation coefficients range from {rfield.min():.3f} to {rfield.max():.3f}")
        print(f"P-values range from {pfield.min():.3f} to {pfield.max():.3f}")

        # Return the correlation coefficients and p-values
        return rfield, pfield

    except Exception as e:
        print(f"Error calculating correlations: {e}")
        sys.exit()

# function for processing the model data for plotting
def process_model_data_for_plot(model_data, models):
    """
    Processes the model data and calculates the ensemble mean.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.

    Returns:
    ensemble_mean (xarray.core.dataarray.DataArray): The equally weighted ensemble mean of the ensemble members.
    """
    # Initialize a list for the ensemble members
    ensemble_members = []

    # Initialize a dictionary to store the number of ensemble members
    ensemble_members_count = {}

    # First constrain the years to the years that are in all of the models
    model_data = constrain_years_processed_hist(model_data, models)

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Print
        print("extracting data for model:", model)

        # Set the ensemble members count to zero
        # if the model is not in the ensemble members count dictionary
        if model not in ensemble_members_count:
            ensemble_members_count[model] = 0
        
        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Append the ensemble member to the list of ensemble members
            ensemble_members.append(member)

            # Try to print values for each member
            # print("trying to print values for each member for debugging")
            # print("values for model:", model)
            # print("values for members:", member)
            # print("member values:", member.values)

            # Extract the lat and lon values
            lat = member.lat.values
            lon = member.lon.values

            # Extract the years
            years = member.time.dt.year.values

            # Print statements for debugging
            # print('shape of years', np.shape(years))
            # # print('years', years)

            # Increment the count of ensemble members for the model
            ensemble_members_count[model] += 1

    # Convert the list of all ensemble members to a numpy array
    ensemble_members = np.array(ensemble_members)

    # Print the dimensions of the ensemble members
    # print("ensemble members shape", np.shape(ensemble_members))


    # Take the equally weighted ensemble mean
    ensemble_mean = ensemble_members.mean(axis=0)

    # Print the dimensions of the ensemble mean
    # print(np.shape(ensemble_mean))
    # print(type(ensemble_mean))
    # print(ensemble_mean)
        
    # Convert ensemble_mean to an xarray DataArray
    ensemble_mean = xr.DataArray(ensemble_mean, coords=member.coords, dims=member.dims)

    return ensemble_mean, lat, lon, years, ensemble_members_count

# Function for calculating the spatial correlations
def calculate_spatial_correlations(observed_data, model_data, models, variable):
    """
    Ensures that the observed and model data have the same dimensions, format and shape. Before calculating the spatial correlations between the two datasets.
    
    Parameters:
    observed_data (xarray.core.dataset.Dataset): The processed observed data.
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.

    Returns:
    rfield (xarray.core.dataarray.DataArray): The spatial correlations between the observed and model data.
    pfield (xarray.core.dataarray.DataArray): The p-values for the spatial correlations between the observed and model data.
    """
    # try:
    # Process the model data and calculate the ensemble mean
    ensemble_mean, lat, lon, years, ensemble_members_count = process_model_data_for_plot(model_data, models)

    # Debug the model data
    # print("ensemble mean within spatial correlation function:", ensemble_mean)
    # print("shape of ensemble mean within spatial correlation function:", np.shape(ensemble_mean))
    
    # Extract the lat and lon values
    obs_lat = observed_data.lat.values
    obs_lon = observed_data.lon.values
    # And the years
    obs_years = observed_data.time.dt.year.values

    # Initialize lists for the converted lons
    obs_lons_converted, lons_converted = [], []

    # Transform the obs lons
    obs_lons_converted = np.where(obs_lon > 180, obs_lon - 360, obs_lon)
    # add 180 to the obs_lons_converted
    obs_lons_converted = obs_lons_converted + 180

    # For the model lons
    lons_converted = np.where(lon > 180, lon - 360, lon)
    # # add 180 to the lons_converted
    lons_converted = lons_converted + 180

    # Print the observed and model years
    # print('observed years', obs_years)
    # print('model years', years)
    
    # Find the years that are in both the observed and model data
    years_in_both = np.intersect1d(obs_years, years)

    # print('years in both', years_in_both)

    # Select only the years that are in both the observed and model data
    observed_data = observed_data.sel(time=observed_data.time.dt.year.isin(years_in_both))
    ensemble_mean = ensemble_mean.sel(time=ensemble_mean.time.dt.year.isin(years_in_both))

    # Remove years with NaNs
    observed_data, ensemble_mean = remove_years_with_nans(observed_data, ensemble_mean, variable)

    # Print the ensemble mean values
    # print("ensemble mean value after removing nans:", ensemble_mean.values)

    # # set the obs_var_name
    # obs_var_name = variable
    
    # # choose the variable name for the observed data
    # # Translate the variable name to the name used in the obs dataset
    # if obs_var_name == "psl":
    #     obs_var_name = "msl"
    # elif obs_var_name == "tas":
    #     obs_var_name = "t2m"
    # elif obs_var_name == "sfcWind":
    #     obs_var_name = "si10"
    # elif obs_var_name == "rsds":
    #     obs_var_name = "ssrd"
    # elif obs_var_name == "tos":
    #     obs_var_name = "sst"
    # else:
    #     print("Invalid variable name")
    #     sys.exit()

    # variable extracted already
    # Convert both the observed and model data to numpy arrays
    observed_data_array = observed_data.values / 100
    ensemble_mean_array = ensemble_mean.values / 100

    # Print the values and shapes of the observed and model data
    # print("observed data shape", np.shape(observed_data_array))
    # print("model data shape", np.shape(ensemble_mean_array))
    # print("observed data", observed_data_array)
    # print("model data", ensemble_mean_array)

    # Check that the observed data and ensemble mean have the same shape
    if observed_data_array.shape != ensemble_mean_array.shape:
        raise ValueError("Observed data and ensemble mean must have the same shape.")

    # Calculate the correlations between the observed and model data
    rfield, pfield = calculate_correlations(observed_data_array, ensemble_mean_array, obs_lat, obs_lon)

    return rfield, pfield, obs_lons_converted, lons_converted, observed_data, ensemble_mean, ensemble_members_count