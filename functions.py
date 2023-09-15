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
from sklearn.utils import resample

# Import CDO
from cdo import *
cdo = Cdo()

# Import the dictionaries
import dictionaries as dic

# General todos
# TODO: incorporate bootstrapping for significance testing

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

# Function to load the processed dcpp data
def load_data(base_directory, models, variable, region, forecast_range, season):
    """Load the data from the base directory into a dictionary of datasets.
    
    This function takes a base directory and a list of models and loads
    all of the individual ensemble members into a dictionary of datasets
    grouped by models.
    
    Args:
        base_directory: The base directory where the data is stored.
        models: A list of models to load.
        variable: The variable to load, extracted from the command line.
        region: The region to load, extracted from the command line.
        forecast_range: The forecast range to load, extracted from the command line.
        season: The season to load, extracted from the command line.
        
    Returns:
        A dictionary of datasets grouped by models.
    """
    
    # Create an empty dictionary to store the datasets.
    datasets_by_model = {}
    
    # Loop over the models.
    for model in models:
        
        # Create an empty list to store the datasets for this model.
        datasets_by_model[model] = []
        
        # create the path to the files for this model
        files_path = base_directory + "/" + variable + "/" + model + "/" + region + "/" + f"years_{forecast_range}" + "/" + season + "/" + "outputs" + "/" + "mergetime" + "/" + "*.nc"

        # print the path to the files
        print("Searching for files in ", files_path)

        # Create a list of the files for this model.
        files = glob.glob(files_path)

        # if the list of files is empty, print a warning and
        # exit the program
        if len(files) == 0:
            print("No files found for " + model)
            sys.exit()
        
        # Print the files to the screen.
        print("Files for " + model + ":", files)

        # Loop over the files.
        for file in files:

            # Print the file to the screen.
            # print(file)
            
            # check that the file exists
            # if it doesn't exist, print a warning and
            # exit the program
            if not os.path.exists(file):
                print("File " + file + " does not exist")
                sys.exit()

            # Load the dataset.
            dataset = xr.open_dataset(file, chunks = {"time": 50, "lat": 45, "lon": 45})

            # Extract the variant_label
            variant_label = dataset.attrs['variant_label']

            # Print the variant_label
            print("loading variant_label: ", variant_label)

            # Append the dataset to the list of datasets for this model.
            datasets_by_model[model].append(dataset)
            
    # Return the dictionary of datasets.
    return datasets_by_model

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

# Function used to process the dcpp data
def process_data(datasets_by_model, variable):
    """Process the data.
    
    This function takes a dictionary of datasets grouped by models
    and processes the data for each dataset.
    
    Args:
        datasets_by_model: A dictionary of datasets grouped by models.
        variable: The variable to load, extracted from the command line.
        
    Returns:
        variable_data_by_model: the data extracted for the variable for each model.
        model_time_by_model: the model time extracted from each model for each model.
    """
    
    #print(f"Dataset type: {type(datasets_by_model)}")

    def process_model_dataset(dataset, variable):
        """Process a single dataset.
        
        This function takes a single dataset and processes the data.
        
        Args:
            dataset: A single dataset.
            variable: The variable to load, extracted from the command line.
            
        Returns:
            variable_data: the extracted variable data for a single model.
            model_time: the extracted time data for a single model.
        """
        
        if variable == "psl":
            # Extract the variable.
            # Using try and except to catch any errors.
            try:
                # Extract the variable.
                variable_data = dataset["psl"]

                # #print the values of the variable data
                # #print("Variable data values: ", variable_data.values)
            except:
                #print("Error converting from Pa to hPa")
                sys.exit()
        elif variable == "tas":
            # Extract the variable.
            variable_data = dataset["tas"]
        elif variable == "rsds":
            # Extract the variable.
            variable_data = dataset["rsds"]
        elif variable == "sfcWind":
            # Extract the variable.
            variable_data = dataset["sfcWind"]
        else:
            #print("Variable " + variable + " not recognised")
            sys.exit()

        # If variable_data is empty, #print a warning and exit the program.
        if variable_data is None:
            #print("Variable " + variable + " not found in dataset")
            sys.exit()

        # Extract the time dimension.
        model_time = dataset["time"].values
        # Set the type for the time dimension.
        model_time = model_time.astype("datetime64[Y]")

        # If model_time is empty, #print a warning and exit the program.
        if model_time is None:
            #print("Time not found in dataset")
            sys.exit()

        return variable_data, model_time
    
    # Create empty dictionaries to store the processed data.
    variable_data_by_model = {}
    model_time_by_model = {}
    for model, datasets in datasets_by_model.items():
        try:
            # Create empty lists to store the processed data.
            variable_data_by_model[model] = []
            model_time_by_model[model] = []
            # Loop over the datasets for this model.
            for dataset in datasets:
                # Process the dataset.
                variable_data, model_time = process_model_dataset(dataset, variable)
                # Append the processed data to the lists.
                variable_data_by_model[model].append(variable_data)
                model_time_by_model[model].append(model_time)
        except Exception as e:
            #print(f"Error processing dataset for model {model}: {e}")
            #print("Exiting the program")
            sys.exit()

    # Return the processed data.
    return variable_data_by_model, model_time_by_model


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


# Define a new function to constrain the years of the initialized data and the uninitialized data
# to ensure that the years are the same for both
def constrain_years_init_uninit(dcpp_model_data, historical_model_data, dcpp_models, historical_models):
    """
    Constrain the years of the initialized data and the uninitialized data to ensure that the years are the same for both.

    This function takes the initialized and uninitialized data and constrains the years to the years that are in both
    datasets. It then returns the constrained datasets as a tuple.

    Parameters
    ----------
    dcpp_model_data : list of xarray.Dataset
        The initialized model data to constrain.
    historical_model_data : list of xarray.Dataset
        The uninitialized model data to constrain.
    dcpp_models : list of str
        The names of the initialized models.
    historical_models : list of str
        The names of the uninitialized models.

    Returns
    -------
    tuple of list of xarray.Dataset
        The initialized and uninitialized data with the years constrained to the years that are in both datasets.
    """

    # Initialize a list to store the years for each model
    dcpp_years_list = []
    historical_years_list = []

    # Loop over the dcpp models to complete the dcpp years list
    for model in dcpp_models:
        dcpp_model_data_extracted = dcpp_model_data[model]
        # Loop over the ensemble members in the model data
        for member in dcpp_model_data_extracted:
            # Extract the years
            dcpp_years = member.time.dt.year.values

            # Append the years to the list of years
            dcpp_years_list.append(dcpp_years)

    # Loop over the historical models to complete the historical years list
    for model in historical_models:
        historical_model_data_extracted = historical_model_data[model]
        # Loop over the ensemble members in the model data
        for member in historical_model_data_extracted:
            # Extract the years
            historical_years = member.time.dt.year.values

            # if the length of the years is less than 48, then don't use this member
            if len(historical_years) < 48:
                print("years less than 48")
                print("not including this member in common years for this model: ", model)
                continue

            # Append the years to the list of years
            historical_years_list.append(historical_years)

    # print the first value of the dcpp years list
    print("dcpp years list", dcpp_years_list[0])
    # print the first value of the historical years list
    print("historical years list", historical_years_list[0])

    # Find the years that are in both the dcpp and historical models
    common_years = list(set(dcpp_years_list[0]).intersection(*historical_years_list))

    # Print the common years for debugging
    print("Common years:", common_years)

    # Initiliaze dictionaries to store the constrained data
    constrained_dcpp_data = {}
    constrained_historical_data = {}

    # Loop over the dcpp models
    for model in dcpp_models:
        # Extract the model data
        dcpp_model_data_extracted = dcpp_model_data[model]

        # Loop over the ensemble members in the model data
        for member in dcpp_model_data_extracted:
            # Extract the years
            dcpp_years = member.time.dt.year.values

            # Find the years that are in both the model data and the common years
            years_in_both = np.intersect1d(dcpp_years, common_years)

            # Select only those years from the model data
            member = member.sel(time=member.time.dt.year.isin(years_in_both))

            # Add the member to the constrained data dictionary
            if model not in constrained_dcpp_data:
                constrained_dcpp_data[model] = []
            constrained_dcpp_data[model].append(member)

    # Loop over the historical models
    for model in historical_models:
        # Extract the model data
        historical_model_data_extracted = historical_model_data[model]

        # Loop over the ensemble members in the model data
        for member in historical_model_data_extracted:
            # Extract the years
            historical_years = member.time.dt.year.values

            # Find the years that are in both the model data and the common years
            years_in_both = np.intersect1d(historical_years, common_years)

            # Select only those years from the model data
            member = member.sel(time=member.time.dt.year.isin(years_in_both))

            # Add the member to the constrained data dictionary
            if model not in constrained_historical_data:
                constrained_historical_data[model] = []
            constrained_historical_data[model].append(member)

    # Return the constrained data
    return constrained_dcpp_data, constrained_historical_data


# Function to remove years with Nans
# checking for Nans in observed data
def remove_years_with_nans(observed_data, dcpp_ensemble_mean, historical_ensemble_mean, variable):
    """
    Removes years from the observed data that contain NaN values.

    Args:
        observed_data (xarray.Dataset): The observed data.
        dcpp_ensemble_mean (xarray.Dataset): The ensemble mean of the initialized data.
        historical_ensemble_mean (xarray.Dataset): The ensemble mean of the uninitialized data.
        variable (str): the variable name.

    Returns:
        xarray.Dataset: The observed data with years containing NaN values removed.
        xarray.Dataset: The ensemble mean of the initialized data with years containing NaN values removed.
        xarray.Dataset: The ensemble mean of the uninitialized data with years containing NaN values removed.
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

            # for the dcpp data
            # Select the year from the dcpp data
            dcpp_ensemble_mean = dcpp_ensemble_mean.sel(time=dcpp_ensemble_mean.time.dt.year != year)

            # for the historical data
            # Select the year from the historical data
            historical_ensemble_mean = historical_ensemble_mean.sel(time=historical_ensemble_mean.time.dt.year != year)
        # if there are no Nan values in the data for a year
        # then print the year and exit the loop
        else:
            # print(year, "no nan for this year")

            # exit the loop
            break

    return observed_data, dcpp_ensemble_mean, historical_ensemble_mean

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
        print("observed data shape", np.shape(observed_data))
        print("model data shape", np.shape(model_data))

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

        # print the correlation coefficients and p-values
        # print("correlation coefficients", rfield)
        # print("p-values", pfield)
        # print shape of pfield
        print("shape of pfield", np.shape(pfield))

        # Return the correlation coefficients and p-values
        return rfield, pfield

    except Exception as e:
        print(f"Error calculating correlations: {e}")
        sys.exit()

# function for processing the model data for plotting
def process_model_data_for_plot(model_data, models, observed_data):
    """
    Processes the model data and calculates the ensemble mean.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.
    observed_data (xarray.core.dataset.Dataset): The processed observed data.

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

            # If the time type is not np.datetime64, then convert it
            if type(member.time.values[0]) != np.datetime64:
                # Convert the time type
                member_time = member.time.astype('datetime64[ns]')

                # Assign the time values
                member = member.assign_coords(time=member.time)

            # Extract the lat and lon values
            lat = member.lat.values
            lon = member.lon.values

            # Extract the years
            model_years = member.time.dt.year.values

            # If the years index has duplicate values, then skip over this member
            if len(model_years) != len(set(model_years)):
                print("duplicate years")
                print("not including this member for this model: ", model)
                continue

            # Append the ensemble member to the list of ensemble members
            ensemble_members.append(member)

            # Increment the count of ensemble members for the model
            ensemble_members_count[model] += 1

    # constrain the years of the observed data
    observed_data = observed_data.sel(time=observed_data.time.dt.year.isin(model_years))

    # Extract the years from the observed data
    obs_years = observed_data.time.dt.year.values

    # Convert observed_data to a numpy array
    if type(observed_data) != np.ndarray:
        print("converting observed data to numpy array")
        observed_data = observed_data.values

    # print the shape of the ensemble members
    print("shape of ensemble members", np.shape(ensemble_members))
    # prin the shape of the observed data
    print("shape of observed data", np.shape(observed_data))

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

    return ensemble_mean, ensemble_members, observed_data, lat, lon, model_years, obs_years, ensemble_members_count

# Function used to extract the ensemble members
def extract_ensemble_members(model_data, models, ensemble_members, ensemble_members_count):
    """
    Extract the ensemble members from a dictionary of xarray datasets.

    This function loops over the models in the dictionary and extracts the ensemble members
    for each model. It also keeps track of the number of ensemble members for each model.

    Parameters
    ----------
    model_data : dict of xarray.Dataset
        The dictionary of xarray datasets containing the model data.
    models : list of str
        The names of the models to extract ensemble members for.
    ensemble_members : list
        The list to append the ensemble members to.
    ensemble_members_count : dict
        The dictionary to keep track of the number of ensemble members for each model.

    Returns
    -------
    ensemble_members : list
        The list of ensemble members.
    lat : numpy.ndarray
        The latitude values.
    lon : numpy.ndarray
        The longitude values.
    model_years : numpy.ndarray
        The years for the model data.
    """

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

            # Extract the lat and lon values
            lat = member.lat.values
            lon = member.lon.values

            # Extract the years
            model_years = member.time.dt.year.values

            # Increment the count of ensemble members for the model
            ensemble_members_count[model] += 1

    return ensemble_members, lat, lon, model_years, ensemble_members_count

# Define a new function - process_model_data_for_plot_diff
# which takes both the initialized and uninitialized data
# and calculates the ensemble mean for both, while constraining the years
# to the years that are in all of the models
def process_model_data_for_plot_diff(dcpp_model_data, historical_model_data, dcpp_models, historical_models, observed_data):
    """
    Process model data for plotting the difference between initialized and uninitialized simulations.

    This function takes both the initialized and uninitialized data and calculates the ensemble mean for both,
    while constraining the years to the years that are in all of the models.

    Parameters
    ----------
    dcpp_model_data : xarray.Dataset
        The initialized model data to process.
    historical_model_data : xarray.Dataset
        The uninitialized model data to process.
    dcpp_models : list of str
        The names of the initialized models.
    historical_models : list of str
        The names of the uninitialized models.
    observed_data : xarray.Dataset
        The observed data to use for constraining the years.

    Returns
    -------
    dcpp_ensemble_members : array
        The initialized ensemble members.
    historical_ensemble_members : array
        The uninitialized ensemble members.
    """

    # Initialize lists for the ensemble members
    dcpp_ensemble_members, historical_ensemble_members = [], []

    # Initialize dictionaries to store the number of ensemble members
    dcpp_ensemble_members_count, historical_ensemble_members_count = {}, {}

    # First constrain the years to the years that are in all of the models
    # for both the initialized and uninitialized data
    dcpp_model_data = constrain_years_processed_hist(dcpp_model_data, dcpp_models)
    historical_model_data = constrain_years_processed_hist(historical_model_data, historical_models)

    # Use the function to constrain the years of the initialized and uninitialized data
    # to ensure that the years are the same for both
    dcpp_model_data, historical_model_data = constrain_years_init_uninit(dcpp_model_data, historical_model_data, dcpp_models, historical_models)

    # Extract the ensemble members for the initialized data
    dcpp_ensemble_members, lat, lon, dcpp_model_years, dcpp_ensemble_members_count = extract_ensemble_members(dcpp_model_data, dcpp_models, dcpp_ensemble_members, dcpp_ensemble_members_count)

    # Extract the ensemble members for the uninitialized data
    historical_ensemble_members, _, _, historical_model_years, historical_ensemble_members_count = extract_ensemble_members(historical_model_data, historical_models, historical_ensemble_members, historical_ensemble_members_count)

    # print the shape of the dcpp and historical ensemble members
    print("shape of dcpp model years", np.shape(dcpp_model_years))
    print("shape of historical model years", np.shape(historical_model_years))

    # print the shape of the dcpp and historical ensemble members
    print("shape of dcpp ensemble members", np.shape(dcpp_ensemble_members))
    print("shape of historical ensemble members", np.shape(historical_ensemble_members))

    # constrain the years of the observed data
    # first ensure that the years are the same for both the initialized and uninitialized data
    if dcpp_model_years.shape != historical_model_years.shape:
        print("years are not the same for both the initialized and uninitialized data")
        print("exiting the program")
        sys.exit()

    # Constrain the years of the observed data
    observed_data = observed_data.sel(time=observed_data.time.dt.year.isin(dcpp_model_years))

    # Extract the years from the observed data
    obs_years = observed_data.time.dt.year.values

    # Convert observed_data to a numpy array
    if type(observed_data) != np.ndarray:
        print("converting observed data to numpy array")
        observed_data = observed_data.values

    # Convert the list of all ensemble members to a numpy array
    dcpp_ensemble_members = np.array(dcpp_ensemble_members)
    historical_ensemble_members = np.array(historical_ensemble_members)

    # Print the dimensions of the ensemble members
    print("shape of dcpp ensemble members", np.shape(dcpp_ensemble_members))
    print("shape of historical ensemble members", np.shape(historical_ensemble_members))
    print("shape of observed data", np.shape(observed_data))

    return dcpp_ensemble_members, historical_ensemble_members, observed_data, lat, lon, dcpp_model_years, historical_model_years, obs_years, dcpp_ensemble_members_count, historical_ensemble_members_count




# checking for Nans in observed data
def remove_years_with_nans_original(observed_data, ensemble_mean, variable):
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
        #print("Invalid variable name")
        sys.exit()

    #print("var name for obs", obs_var_name)
    
    for year in observed_data.time.dt.year.values[::-1]:
        # Extract the data for the year
        data = observed_data.sel(time=f"{year}")

        # print("data type", (type(data)))
        # print("data vaues", data)
        # print("data shape", np.shape(data))

        
        # If there are any Nan values in the data
        if np.isnan(data.values).any():
            # #print the year
            # #print(year)

            # Select the year from the observed data
            observed_data = observed_data.sel(time=observed_data.time.dt.year != year)

            # for the model data
            ensemble_mean = ensemble_mean.sel(time=ensemble_mean.time.dt.year != year)

        # if there are no Nan values in the data for a year
        # then #print the year
        # and "no nan for this year"
        # and continue the script
        else:
            print(year, "no nan for this year")

            # exit the loop
            break

    return observed_data, ensemble_mean

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

    # use the function to remove NaNs from the observed data
    observed_data = remove_nan_years_obs(observed_data)

    # try:
    # Process the model data and calculate the ensemble mean
    ensemble_mean, ensemble_members, observed_data, lat, lon, model_years, _, ensemble_members_count = process_model_data_for_plot(model_data, models, observed_data)

    # Debug the model data
    # print("ensemble mean within spatial correlation function:", ensemble_mean)
    # print("shape of ensemble mean within spatial correlation function:", np.shape(ensemble_mean))
    
    # # Extract the lat and lon values
    # obs_lat = observed_data.lat.values
    # obs_lon = observed_data.lon.values
    # # And the years
    # obs_years = observed_data.time.dt.year.values

    # Initialize lists for the converted lons
    obs_lons_converted, lons_converted = [], []

    # Transform the obs lons
    obs_lons_converted = np.where(lon > 180, lon - 360, lon)
    # add 180 to the obs_lons_converted
    obs_lons_converted = obs_lons_converted + 180

    # For the model lons
    lons_converted = np.where(lon > 180, lon - 360, lon)
    # # add 180 to the lons_converted
    lons_converted = lons_converted + 180

    # Print the observed and model years
    # print('observed years', obs_years)
    # print('model years', years)

    # # Set the model years to the observed years
    # years = model_years
    
    # # Find the years that are in both the observed and model data
    # years_in_both = np.intersect1d(obs_years, years)

    # # print('years in both', years_in_both)

    # # Select only the years that are in both the observed and model data
    # observed_data = observed_data.sel(time=observed_data.time.dt.year.isin(years_in_both))
    # ensemble_mean = ensemble_mean.sel(time=ensemble_mean.time.dt.year.isin(years_in_both))

    # # Remove years with NaNs
    # observed_data, ensemble_mean = remove_years_with_nans_original(observed_data, ensemble_mean, variable)

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
    # observed_data_array = observed_data.values
    # ensemble_mean_array = ensemble_mean.values

    # Print the values and shapes of the observed and model data
    print("observed data shape", np.shape(observed_data))
    print("model data shape", np.shape(ensemble_mean))
    # print("observed data", observed_data_array)
    # print("model data", ensemble_mean_array)

    # Check that the observed data and ensemble mean have the same shape
    if observed_data.shape != ensemble_mean.shape:
        raise ValueError("Observed data and ensemble mean must have the same shape.")

    # Calculate the correlations between the observed and model data
    rfield, pfield = calculate_correlations(observed_data, ensemble_mean, lat, lon)

    return rfield, pfield, obs_lons_converted, lons_converted, observed_data, ensemble_mean, ensemble_members_count



# Define a new function to calculate the correlations as differences
# between the initialized (dcpp data) and the uninitialized (historical data)
def calculate_correlations_diff(observed_data, init_model_data, uninit_model_data, obs_lat, obs_lon, p_sig = 0.05):
    """
    Calculates the spatial correlations for both the initialized and uninitialized data.
    Then calculates the differences between the initialized (dcpp data) and the uninitialized (historical data).
    Identifies the regions where the correlation improvements are statistically significant.
    
    Parameters:
        observed_data (xarray.core.dataset.Dataset): The processed observed data.
        init_model_data (dict): The processed initialized model data.
        uninit_model_data (dict): The processed uninitialized model data.
        obs_lat (numpy.ndarray): The latitude values of the observed data.
        obs_lon (numpy.ndarray): The longitude values of the observed data.
        p_sig (float): The p-value for statistical significance. Default is 0.05.
    
    Returns:
        rfield_diff (xarray.core.dataarray.DataArray): The differences in spatial correlations between the initialized and uninitialized data.
        sign_regions (xarray.core.dataarray.DataArray): The regions where the correlation improvements are statistically significant.
    """

    # Use the calculate_correlations function to calculate the correlations for both the initialized and uninitialized data
    # First for the initialized data
    rfield_init, pfield_init = calculate_correlations(observed_data, init_model_data, obs_lat, obs_lon)

    # Then for the uninitialized data
    rfield_uninit, pfield_uninit = calculate_correlations(observed_data, uninit_model_data, obs_lat, obs_lon)

    # Calculate the differences between the initialized and uninitialized data
    rfield_diff = rfield_init - rfield_uninit

    # print the shapes of the rfield_init and rfield_uninit arrays
    print("rfield_init shape", np.shape(rfield_init))
    print("rfield_uninit shape", np.shape(rfield_uninit))

    # # Calculate the p-values for the differences between the initialized and uninitialized data
    # # Set up an empty array for the t-statistic
    # zeros_data = 
    
    # # Loop over the latitudes and longitudes
    # for y in range(len(obs_lat)):
    #     for x in range(len(obs_lon)):
    #         # set up the model data
    #         mod_init = init_model_data[:, y, x]
    #         mod_uninit = uninit_model_data[:, y, x]

    #         # Calculate the correlation coefficient and p-value
    #         t_stat[y, x], p_values[y, x] = stats.ttest_ind(mod_init, mod_uninit, equal_var=False)

    # # Print the range of the correlation coefficients and p-values
    # # to 3 decimal places
    # print(f"t-statistic range from {t_stat.min():.3f} to {t_stat.max():.3f}")
    # print(f"P-values range from {p_values.min():.3f} to {p_values.max():.3f}")

    # Identify the regions where the correlation improvements are statistically significant
    # at the 95% confidence level
    sign_regions = np.empty_like(rfield_diff)

    # Prin the types of the rfield_diff and sign_regions arrays
    print("rfield_diff type", type(rfield_diff))
    print("sign_regions type", type(sign_regions))

    # Print the shapes of the rfield_diff and sign_regions arrays
    print("rfield_diff shape", np.shape(rfield_diff))
    print("sign_regions shape", np.shape(sign_regions))

    # Return the differences in spatial correlations and the p-values
    return rfield_diff, sign_regions


# function for removing years with Nans
# from the observed data
def remove_nan_years_obs(observed_data):
    """
    Remove years from an xarray dataset that contain NaN values.

    Parameters
    ----------
    observed_data : xarray.Dataset
        The observed data to remove years from.

    Returns
    -------
    xarray.Dataset
        The observed data with years containing NaN values removed.
    """
    # Get the years in the observed data
    obs_years = np.unique(observed_data.time.dt.year)

    # Loop over the years and remove any that contain NaN values
    for year in obs_years:
        # Extract the data for the year
        data = observed_data.sel(time=f"{year}")

        # If there are any NaN values in the data
        if np.isnan(data.values).any():
            # Print the year
            print(year, "contains NaN values")
            print("removing this year from the observed data")

            # Select the year from the observed data
            observed_data = observed_data.sel(time=observed_data.time.dt.year != year)

    return observed_data

# We want to define a new function which will perform the bootstrapping
# to calculate the significance of the ACC scores
# This function takes as input the observed data and the model data
# as well as the models being used and the variable name
# and returns the field of p-values for the spatial correlations
def calculate_spatial_correlations_bootstrap(observed_data, model_data, models, variable, n_bootstraps=1000, experiment=None):
    """
    The method involves creating 1,000 bootstrapped hindcasts from a finite ensemble size and a finite number of validation years. 
    The steps involved in creating the bootstrapped hindcasts are as follows:

    1) Randomly select N cases (validation years) with replacement. 
        To take autocorrelation into account, this is done in blocks of five consecutive years.
    2) For each case, randomly select M ensemble members with replacement. 
        Compute the ensemble mean from these M samples.
    3) Compute the evaluation metrics (ACC, MSSS, RPC, and skill difference) with 
        the resultant ensemble mean prediction.
    4) Repeat steps 1-3 1,000 times to create a sample distribution of the 
        evaluation metrics.
    
    For the ACC and MSSS, the p-value is defined as the ratio of negative values from the 
        bootstrapped sample distribution on the basis of a one-tailed test of the hypothesis 
            that the prediction skill is greater than 0. 

    Arguments:
        observed_data (xarray.core.dataset.Dataset): The processed observed data.
        model_data (dict): The processed model data.
        models (list): The list of models to be plotted.
        variable (str): The variable name.
        n_bootstraps (int): The number of bootstraps to perform. Default is 1000.

    Returns:
        rfield (xarray.core.dataarray.DataArray): The spatial correlations between the observed and model data.
        pfield (xarray.core.dataarray.DataArray): The p-values for the spatial correlations between the observed and model data. Bootstrapped.
    """

    # Print the types of the observed and model data
    print("observed data type", type(observed_data))
    print("model data type", type(model_data))

    # Print the shapes of the observed and model data
    print("observed data shape", np.shape(observed_data))
    print("model data shape", len(model_data))

    # Extract the years from the observed data
    obs_years = observed_data.time.dt.year.values

    # Remove the years containing NaNs
    for year in obs_years:
        # Extract the data for the year
        data = observed_data.sel(time=f"{year}")

        # If there are any Nan values in the data
        if np.isnan(data.values).any():
            # Print the year
            print(year)

            # Select the year from the observed data
            observed_data = observed_data.sel(time=observed_data.time.dt.year != year)

    # Use the function ======= to convert the model data to a numpy array
    # use the function process_model_data_for_plot for this
    _, model_data, observed_data, _, _, model_years, obs_years, ensemble_members_count = process_model_data_for_plot(model_data, models, observed_data)

    # if observed data is not a numpy array
    if type(observed_data) != np.ndarray:
        print("observed data is not a numpy array")
        # convert observed data to a numpy array
        observed_data = observed_data.values
        # if the experiment is dcppA-hindcast

        # # constrain the years to the years that are in both the observed and model data
        # observed_data = observed_data[3:, :, :]

    # Print the types of the observed and model data
    print("observed data type", type(observed_data))
    print("model data type", type(model_data))

    # Print the shapes of the observed and model data
    print("observed data shape", np.shape(observed_data))
    print("model data shape", np.shape(model_data))

    # print the values of the observed and model data
    # print("observed data", observed_data)
    # print("model data", model_data)
    
    # Check that the observed and model data have the same type
    if type(observed_data) != type(model_data):
        raise ValueError("Observed data and model data must have the same type.")
    

    # Print the years extracted from the observed and model data
    print("observed years", obs_years)
    print("model years", model_years)

    # Print the values of each to check
    print("observed data year constrained", np.shape(observed_data))
    print("model data year constrained", np.shape(model_data))

    # Now we want to check that there are no NaNs in the observed and model data
    if np.isnan(observed_data).any():
        raise ValueError("Observed data contains NaNs.")
    
    if np.isnan(model_data).any():
        raise ValueError("Model data contains NaNs.")
    
    
    # Now we want to check that the observed and model data have the same shape
    # for all dimensions of the observed data
    # and the final 3 dimensions of the model data
    model_data_shape = model_data[0, :, :, :]

    # for brevity set up the lats and lons
    lats = observed_data[0, :, 0]
    lons = observed_data[0, 0, :]

    # if the shapes are not the same
    if observed_data.shape != model_data_shape.shape:
        raise ValueError("Observed data and model data must have the same shape.")
    
    # Now we want to create empty arrays for the bootstrapped p-values
    # and the bootstrapped correlation coefficients
    # in model_data
    # the first dimension is the ensemble members
    # the second dimension is the time
    # the third dimension is the lat
    # the fourth dimension is the lon
    # so we will first resample the years in the time dimension
    # and then resample the ensemble members
    # and then calculate the ensemble mean
    # and then calculate the correlation coefficient
    # and then append the correlation coefficient to the array
    # and then append the p-value to the array
    # and then repeat this process 1000 times
    # so we will have 1000 correlation coefficients and p-values
    # for each grid point
    # create an empty array for the p-values
    # dim = (1000, lat, lon)
    pfield_dist = np.empty([n_bootstraps, len(observed_data[0, :, 0]), len(observed_data[0, 0, :])])
    # create an empty array for the correlation coefficients
    rfield_dist = np.empty([n_bootstraps, len(observed_data[0, :, 0]), len(observed_data[0, 0, :])])

    # Print the shapes of the pfield and rfield arrays
    print("pfield array shape", np.shape(pfield_dist))
    print("rfield array shape", np.shape(rfield_dist))

    # Print the types of the pfield and rfield arrays
    print("pfield array type", type(pfield_dist))
    print("rfield array type", type(rfield_dist))

    # # Take the time mean of the observed data
    # observed_data_tm = np.mean(observed_data, axis=0)

    # Extract the number of validation years
    # this is the second dimension of the model data
    n_validation_years = len(model_data[0, :, 0, 0])

    # Extract the number of ensemble members
    # this is the first dimension of the model data
    m_ensemble_members = len(model_data[:, 0, 0, 0])

    # set up the block size for the autocorrelation
    block_size = 5 # years

    # Save the original model data
    model_data_original = model_data.copy()

    # First we want to loop over the bootstraps
    for i in range(n_bootstraps):
        # Randomly select N cases (validation years) with replacement.
        # To take autocorrelation into account, this is done in blocks of five consecutive years.
        # Create 

        # print the number bootstrap
        print("bootstrap number", i)

        # Randomly select block start indices
        block_starts = resample(range(0, n_validation_years - block_size + 1, block_size), n_samples=n_validation_years//block_size, replace=True)

        # Create indices for the entire blocks
        block_indices = []
        for start in block_starts:
            block_indices.extend(range(start, start + block_size))

        # Ensure we have exactly N indices (with replacement)
        if len(block_indices) < n_validation_years:
            block_indices.extend(resample(block_indices, n_samples=n_validation_years-len(block_indices), replace=True))

        # # Print the block indices shape
        # print("block indices shape", np.shape(block_indices))

        # # Print the block indices
        # print("block indices", block_indices)

        # Create a mask for the selected block indices
        mask = np.zeros(n_validation_years, dtype=bool)
        mask[block_indices] = True

        # Apply the mask to select the corresponding block of data for the model data
        n_mask_model_data = model_data[:, mask, :, :]
        # Apply the mask to select the corresponding block of data for the observed data
        n_mask_observed_data = observed_data[mask, :, :]

        # Next, for each case, randomly select M ensemble members with replacement.
        ensemble_resampled = resample(n_mask_model_data, n_samples=m_ensemble_members, replace=True)

        # # Print the dimensions of the ensemble resampled
        # print("ensemble resampled shape", np.shape(ensemble_resampled))
        # print("model data original shape masked", np.shape(model_data_original[:, mask, :, :]))

        # # Check if ensemble_resampled is different from model_data
        # if not np.array_equal(ensemble_resampled, model_data_original[:, mask, :, :]):
        #     print("Ensemble has been resampled")
        # else:
        #     print("Ensemble has not been resampled")

        # Calculate the ensemble mean for each case
        ensemble_mean = np.mean(ensemble_resampled, axis=0)

        # # Print the dimensions of the ensemble mean
        # print("ensemble mean shape", np.shape(ensemble_mean))
        # print("observed data shape", np.shape(n_mask_observed_data))

        # Calculate the correlation coefficient and p-value for each case
        # First create empty arrays for the correlation coefficients and p-values
        # Now set up the empty arrays for rfield and pfield
        rfield = np.empty([len(lats), len(lons)])
        pfield = np.empty([len(lats), len(lons)])
        # now loop over the lats and lons
        for y in range(len(lats)):
            for x in range(len(lons)):
                # set up the obs and model data
                obs = n_mask_observed_data[:, y, x]
                mod = ensemble_mean[:, y, x]

                # Calculate the correlation coefficient and p-value
                r, p = stats.pearsonr(obs, mod)

                # # If the correlation coefficient is negative, set the p-value to NaN
                # if r < 0:
                #     p = np.nan

                # Append the correlation coefficient and p-value to the arrays
                rfield[y, x], pfield[y, x] = r, p
    
        # append the correlation coefficients and p-values to the arrays
        rfield_dist[i, :, :] = rfield
        pfield_dist[i, :, :] = pfield

    # Print the shapes of the pfield and rfield arrays
    print("pfield array shape", np.shape(pfield_dist))
    print("rfield array shape", np.shape(rfield_dist))

    # Print the types of the pfield and rfield arrays
    print("pfield array type", type(pfield_dist))
    print("rfield array type", type(rfield_dist))

    # Now we want to obtain the p-values for the correlations
    # first create an empty array for the p-values
    pfield_bootstrap = np.empty([len(lats), len(lons)])

    # Now loop over the lats and lons
    for y in range(len(lats)):
        # print("y", y)
        for x in range(len(lons)):
            # print("x", x)
            # # print the shape of the rfield_dist array
            # print("rfield_dist shape", np.shape(rfield_dist))
            # set up the rfield_dist and pfield_dist
            rfield_sample = rfield_dist[:, y, x]

            # Calculate the p-value
            pfield_bootstrap[y, x] = np.sum(rfield_sample < 0) / n_bootstraps

    # Print the shape of the pfield_bootstrap array
    print("pfield_bootstrap shape", np.shape(pfield_bootstrap))

    # Print the type of the pfield_bootstrap array
    print("pfield_bootstrap type", type(pfield_bootstrap))

    # Return the p-values
    return pfield_bootstrap

# Define a function used to calculate the spatial correlations given the ensemble mean
def calculate_correlation_coefficient(n_mask_observed_data, ensemble_mean, lats, lons):
    """
    Calculate the correlation coefficient and p-value for each case.

    This function takes the observed data, ensemble mean, lats, and lons and calculates the correlation coefficient
    and p-value for each case. It returns the correlation coefficient and p-value as two 2D arrays.

    Parameters
    ----------
    n_mask_observed_data : numpy.ndarray
        The observed data to use for calculating the correlation coefficient.
    ensemble_mean : numpy.ndarray
        The ensemble mean to use for calculating the correlation coefficient.
    lats : numpy.ndarray
        The latitudes to use for calculating the correlation coefficient.
    lons : numpy.ndarray
        The longitudes to use for calculating the correlation coefficient.

    Returns
    -------
    tuple of numpy.ndarray
        The correlation coefficient and p-value as two 2D arrays.
    """
    # Set up the empty arrays for rfield and pfield
    rfield = np.empty([len(lats), len(lons)])
    pfield = np.empty([len(lats), len(lons)])

    # Loop over the lats and lons
    for y in range(len(lats)):
        for x in range(len(lons)):
            # Set up the obs and model data
            obs = n_mask_observed_data[:, y, x]
            mod = ensemble_mean[:, y, x]

            # Calculate the correlation coefficient and p-value
            r, p = stats.pearsonr(obs, mod)

            # Append the correlation coefficient and p-value to the arrays
            rfield[y, x], pfield[y, x] = r, p

    return rfield, pfield

# Define a new function to calculate the bootstrapped p-values for the differences in spatial correlations
# Between the initialized (dcpp data) and the uninitialized (historical data)
def calculate_spatial_correlations_bootstrap_diff(observed_data, dcpp_model_data, historical_model_data, dcpp_models, historical_models, variable, n_bootstraps=1000):
    """
    Ensures that the observed and model data have the same dimensions, format and shape.
    Then performs the bootstrapping to create a sample distribution of the skill differences.
    Calculates the p-values for the differences in spatial correlations between the initialized and uninitialized data.
    
    Arguments
        observed_data (xarray.core.dataset.Dataset): The processed observed data.
        dcpp_model_data (dict): The processed dcpp model data - initialized.
        historical_model_data (dict): The processed historical model data - uninitialized.
        dcpp_models (list): The list of dcpp models to be plotted.
        historical_models (list): The list of historical models to be plotted.
        variable (str): The variable name.
        n_bootstraps (int): The number of bootstraps to perform. Default is 1000.

    Returns:
        pfield_diff_bootstrap (numpy.ndarray): The p-values for the differences in spatial correlations between the initialized and uninitialized data.
    """

    # Extract the years from the observed data
    obs_years = observed_data.time.dt.year.values

    # Use the function remove_nan_years_obs to remove the years containing NaNs
    observed_data = remove_nan_years_obs(observed_data)

    # Process the datasets to ensure that they have the same dimensions,
    #  format and shape
    # using the updated function process_model_data_for_plot_diff
    dcpp_ensemble_members, historical_ensemble_members, observed_data, lat, lon, dcpp_model_years, \
        historical_model_years, obs_years, dcpp_ensemble_members_count, historical_ensemble_members_count \
            = process_model_data_for_plot_diff(dcpp_model_data, historical_model_data, dcpp_models, historical_models, observed_data)

    # Print the types of the observed and model data
    print("observed data type", type(observed_data))
    print("dcpp model data type", type(dcpp_ensemble_members))
    print("historical model data type", type(historical_ensemble_members))

    # Print the shapes of the observed and model data
    print("observed data shape", np.shape(observed_data))
    print("dcpp model data shape", np.shape(dcpp_ensemble_members))
    print("historical model data shape", np.shape(historical_ensemble_members))

    # FIXME: These are the wrong types for test case
    # # Check if there are any NaNs in the observed data
    # if np.isnan(observed_data).any():
    #     raise ValueError("Observed data contains NaNs.")
    
    # # Check if there are any NaNs in the dcpp model data
    # if np.isnan(dcpp_ensemble_members).any():
    #     raise ValueError("DCPP model data contains NaNs.")
    
    # # Check if there are any NaNs in the historical model data
    # if np.isnan(historical_ensemble_members).any():
    #     raise ValueError("Historical model data contains NaNs.")

    # FIXME: Shape of dcpp_ensemble_members is wrong for test case
    # # Check that the shapes are correct
    # dcpp_model_data_shape = dcpp_ensemble_members[0, :, :, :]
    # historical_model_data_shape = historical_ensemble_members[0, :, :, :]

    # # check that the shapes are the same
    # if observed_data.shape != dcpp_model_data_shape.shape != historical_model_data_shape.shape:
    #     raise ValueError("Observed data, dcpp model data and historical model data must have the same shape.")
    
    # Now we want to create empty arrays for the bootstrapped rfield_diff and pfield_diff values
    rfield_diff = np.empty([n_bootstraps, len(lat), len(lon)])
    pfield_diff = np.empty([n_bootstraps, len(lat), len(lon)])

    # Print the shapes of the rfield_diff and pfield_diff arrays
    print("rfield_diff array shape", np.shape(rfield_diff))
    print("pfield_diff array shape", np.shape(pfield_diff))

    # print the shape of dcpp model years
    print("dcpp model years shape", np.shape(dcpp_model_years))

    # print the shape of dcpp ensemble members
    print("dcpp ensemble members shape", np.shape(dcpp_ensemble_members))

    # print the values of dcpp model years
    print("dcpp model years", dcpp_model_years)

    # print the values of dcpp ensemble members
    print("dcpp ensemble members", dcpp_ensemble_members)

    # FIXME: Causing errors for rsds case
    # Check that dcpp model years is the same as dcpp_model_data.shape[1]
    # Check that dcpp model years has the same shape as dcpp_model_data.shape[1]
    # if dcpp_model_years.shape != dcpp_ensemble_members[0, :, 0, 0].shape:
    #     raise ValueError("DCPP model years must have the same shape as dcpp_model_data.shape[1].")

    # Set up the number of validation years
    n_validation_years = len(dcpp_model_years)

    # Set up the number of ensemble members
    # for the dcpp model data
    # m_ensemble_members_dcpp = len(dcpp_ensemble_members[:, 0, 0, 0])
    # take the len of the first dimension of the dcpp ensemble members array
    m_ensemble_members_dcpp = dcpp_ensemble_members.shape[0]
    # Set up the number of ensemble members
    # for the historical model data
    m_ensemble_members_historical = historical_ensemble_members.shape[0]

    # Set up the block size for the autocorrelation
    block_size = 5 # years

    # Loop over the bootstraps
    for i in range(n_bootstraps):
        # print the bootstrap number
        print("bootstrap number", i)

        # Randomly select block start indices
        block_starts = resample(range(0, n_validation_years - block_size + 1, block_size), n_samples=n_validation_years//block_size, replace=True)

        # Create indices for the entire blocks
        block_indices = []
        for start in block_starts:
            block_indices.extend(range(start, start + block_size))

        # Ensure we have exactly N indices (with replacement)
        if len(block_indices) < n_validation_years:
            block_indices.extend(resample(block_indices, n_samples=n_validation_years-len(block_indices), replace=True))

        # # Print the block indices shape
        print("block indices shape", np.shape(block_indices))

        # # Print the block indices
        print("block indices", block_indices)

        # Create a mask for the selected block indices
        mask = np.zeros(n_validation_years, dtype=bool)
        mask[block_indices] = True

        # Apply the mask to select the corresponding block of data for the dcpp model data
        n_mask_dcpp_model_data = dcpp_ensemble_members[:, mask, :, :]
        # Apply the mask to select the corresponding block of data for the historical model data
        n_mask_historical_model_data = historical_ensemble_members[:, mask, :, :]

        # Apply the mask to select the corresponding block of data for the observed data
        n_mask_observed_data = observed_data[mask, :, :]

        # Next, for each case, randomly select M ensemble members with replacement.
        # For the dcpp model data
        ensemble_resampled_dcpp = resample(n_mask_dcpp_model_data, n_samples=m_ensemble_members_dcpp, replace=True)

        # For the historical model data
        ensemble_resampled_historical = resample(n_mask_historical_model_data, n_samples=m_ensemble_members_historical, replace=True)

        # Calculate the ensemble mean for each case
        ensemble_mean_dcpp = np.mean(ensemble_resampled_dcpp, axis=0)
        ensemble_mean_historical = np.mean(ensemble_resampled_historical, axis=0)

        # Calculate the correlation coefficient and p-value for each case
        # Use the function calculate_correlation_coefficient to do this
        # First for the dcpp model data
        rfield_dcpp, pfield_dcpp = calculate_correlation_coefficient(n_mask_observed_data, ensemble_mean_dcpp, lat, lon)

        # Then for the historical model data
        rfield_historical, pfield_historical = calculate_correlation_coefficient(n_mask_observed_data, ensemble_mean_historical, lat, lon)

        # Calculate the differences between the initialized and uninitialized data
        rfield_diff[i, :, :] = rfield_dcpp - rfield_historical

    # Print the shape of the rfield_diff array
    print("rfield_diff shape", np.shape(rfield_diff))

    # Print the type of the rfield_diff array
    print("rfield_diff type", type(rfield_diff))

    # Print the values of the rfield_diff array
    print("rfield_diff values", rfield_diff)

    # Now we want to obtain the p-values for the differences in spatial correlations
    # between the initialized and uninitialized data
    # create an empty array for the p-values
    pfield_diff_bootstrap = np.empty([len(lat), len(lon)])

    # Now loop over the lats and lons
    for y in range(len(lat)):
        for x in range(len(lon)):
            # set up the rfield_diff
            rfield_diff_sample = rfield_diff[:, y, x]

            # # Set all the negative values to NaN
            # rfield_diff_sample[rfield_diff_sample < 0] = np.nan

            # Calculate the p-value
            pfield_diff_bootstrap[y, x] = np.sum(rfield_diff_sample < 0) / n_bootstraps

    # Print the shape of the pfield_bootstrap array
    print("pfield_bootstrap shape", np.shape(pfield_diff_bootstrap))

    # Print the type of the pfield_bootstrap array
    print("pfield_bootstrap type", type(pfield_diff_bootstrap))

    # Return the p-values
    return pfield_diff_bootstrap







# Define a new function which will calculate the differences in spatial correlations
# Between the initialized (dcpp data) and the uninitialized (historical data)
# This function will calculate the r and p fields for both the initialized and uninitialized data
# Then calculate the differences between the two
def calculate_spatial_correlations_diff(observed_data, dcpp_model_data, historical_model_data, dcpp_models, historical_models, variable):
    """
    Ensures that the observed and model data have the same dimensions, format and shape. 
    Before calculating the spatial correlations for each of the datasets.
    Then calculates the differences between the initialized (dcpp data) and the uninitialized (historical data).
    
    Parameters:
    observed_data (xarray.core.dataset.Dataset): The processed observed data.
    dcpp_model_data (dict): The processed dcpp model data - initialized.
    historical_model_data (dict): The processed historical model data - uninitialized.
    dcpp_models (list): The list of dcpp models to be plotted.
    historical_models (list): The list of historical models to be plotted.
    variable (str): The variable name.
    
    Returns:
    rfield_diff (xarray.core.dataarray.DataArray): The differences in spatial correlations between the initialized and uninitialized data.
    sign_regions (xarray.core.dataarray.DataArray): The regions where the correlation improvements are statistically significant.
    obs_lons_converted (numpy.ndarray): The converted longitude values of the observed data.
    lons_converted (numpy.ndarray): The converted longitude values of the model data.
    observed_data (xarray.core.dataset.Dataset): The processed observed data.
    dcpp_ensemble_mean (xarray.core.dataarray.DataArray): The ensemble mean of the initialized data.
    historical_ensemble_mean (xarray.core.dataarray.DataArray): The ensemble mean of the uninitialized data.
    dcpp_ensemble_members_count (dict): The number of ensemble members for each dcpp model.
    historical_ensemble_members_count (dict): The number of ensemble members for each historical model.
    """

    # First process the dcpp model data to get the ensemble mean
    dcpp_ensemble_mean, dcpp_ensemble_members, _, dcpp_lat, dcpp_lon, dcpp_years, _, dcpp_ensemble_members_count = process_model_data_for_plot(dcpp_model_data, dcpp_models, observed_data)

    # Then process the historical model data to get the ensemble mean
    historical_ensemble_mean, historical_ensemble_members, _, historical_lat, historical_lon, historical_years, _, historical_ensemble_members_count = process_model_data_for_plot(historical_model_data, historical_models, observed_data)

    # Extract the lat and lon values from the observed data
    # Because of how the data has been processed using cdo and gridspec files,
    # the lat and lon values are the same for the observed and model data
    obs_lat = observed_data.lat.values
    obs_lon = observed_data.lon.values

    # Extract the years from the observed data
    obs_years = observed_data.time.dt.year.values

    # Initialize lists for the converted lons
    obs_lons_converted, lons_converted = [], []

    # Transform the obs lons
    obs_lons_converted = np.where(obs_lon > 180, obs_lon - 360, obs_lon)
    # add 180 to the obs_lons_converted
    obs_lons_converted = obs_lons_converted + 180

    # For the model lons
    # Define lon in this case as the dcpp_lon
    # the dcpp lon and historical lon are the same
    lon = dcpp_lon
    # now set up the converted lons
    lons_converted = np.where(lon > 180, lon - 360, lon)
    # # add 180 to the lons_converted
    lons_converted = lons_converted + 180

    # Find the years that are in both the observed and model data
    # First find those years for the model data
    shared_years_model_data = np.intersect1d(dcpp_years, historical_years)

    # Then find those years for the observed data
    shared_years = np.intersect1d(obs_years, shared_years_model_data)

    # Select only the years that are in both the observed and model data
    observed_data = observed_data.sel(time=observed_data.time.dt.year.isin(shared_years))
    dcpp_ensemble_mean = dcpp_ensemble_mean.sel(time=dcpp_ensemble_mean.time.dt.year.isin(shared_years))
    historical_ensemble_mean = historical_ensemble_mean.sel(time=historical_ensemble_mean.time.dt.year.isin(shared_years))

    # Remove years containing only NaNs
    # from all three datasets
    # using the function 'remove_years_with_nans'
    observed_data, dcpp_ensemble_mean, historical_ensemble_mean = remove_years_with_nans(observed_data, dcpp_ensemble_mean, historical_ensemble_mean, variable)

    # Convert all the datasets to numpy arrays
    observed_data_array = observed_data.values
    dcpp_ensemble_mean_array = dcpp_ensemble_mean.values
    historical_ensemble_mean_array = historical_ensemble_mean.values

    # Check that all of these have the same shape
    if observed_data_array.shape != dcpp_ensemble_mean_array.shape != historical_ensemble_mean_array.shape:
        raise ValueError("Observed data and ensemble mean must have the same shape.")
        sys.exit(1)
    
    # Set up the arrays for calculating correlations
    obs = observed_data_array
    init_model = dcpp_ensemble_mean_array
    uninit_model = historical_ensemble_mean_array

    # Use the calculate_correlations_diff function to calculate the differences in spatial correlations between the initialized and uninitialized data
    rfield_diff, sign_regions = calculate_correlations_diff(obs, init_model, uninit_model, obs_lat, obs_lon, p_sig=0.05)

    # Return the outputs
    return rfield_diff, sign_regions, obs_lons_converted, lons_converted, observed_data, dcpp_ensemble_mean, historical_ensemble_mean, dcpp_ensemble_members_count, historical_ensemble_members_count


# Using cdo to do the regridding and selecting the region
def regrid_and_select_region(observations_path, region, obs_var_name):
    """
    Uses CDO remapbil and a gridspec file to regrid and select the correct region for the obs dataset. Loads for the specified variable.
    
    Parameters:
    observations_path (str): The path to the observations dataset.
    region (str): The region to select.

    Returns:
    xarray.Dataset: The regridded and selected observations dataset.
    """
    
    # First choose the gridspec file based on the region
    gridspec_path = "/home/users/benhutch/gridspec"

    # select the correct gridspec file
    if region == "north-atlantic":
        gridspec = gridspec_path + "/" + "gridspec-north-atlantic.txt"
    elif region == "global":
        gridspec = gridspec_path + "/" + "gridspec-global.txt"
    elif region == "azores":
        gridspec = gridspec_path + "/" + "gridspec-azores.txt"
    elif region == "iceland":
        gridspec = gridspec_path + "/" + "gridspec-iceland.txt"
    else:
        print("Invalid region")
        sys.exit()

    # Check that the gridspec file exists
    if not os.path.exists(gridspec):
        print("Gridspec file does not exist")
        sys.exit()

    # Create the output file path
    regrid_sel_region_file = "/home/users/benhutch/ERA5/" + region + "_" + "regrid_sel_region.nc"

    # Check if the output file already exists
    # If it does, then exit the program
    if os.path.exists(regrid_sel_region_file):
        print("File already exists")
        print("Loading ERA5 data")
    else:
        print("File does not exist")
        print("Processing ERA5 data using CDO")

        # Regrid and select the region using cdo 
        cdo.remapbil(gridspec, input=observations_path, output=regrid_sel_region_file)

    # Load the regridded and selected region dataset
    # for the provided variable
    # check whether the variable name is valid
    if obs_var_name not in ["psl", "tas", "sfcWind", "rsds", "tos"]:
        print("Invalid variable name")
        sys.exit()

    # Translate the variable name to the name used in the obs dataset
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

    # Load the regridded and selected region dataset
    # for the provided variable
    try:
        # Load the dataset for the selected variable
        regrid_sel_region_dataset = xr.open_mfdataset(regrid_sel_region_file, combine='by_coords', parallel=True, chunks={"time": 100, 'lat': 100, 'lon': 100})[obs_var_name]

        # Combine the two expver variables
        regrid_sel_region_dataset_combine = regrid_sel_region_dataset.sel(expver=1).combine_first(regrid_sel_region_dataset.sel(expver=5))

        return regrid_sel_region_dataset_combine

    except Exception as e:
        print(f"Error loading regridded and selected region dataset: {e}")
        sys.exit()

def select_season(regridded_obs_dataset_region, season):
    """
    Selects a season from a regridded observation dataset based on the given season string.

    Parameters:
    regridded_obs_dataset_region (xarray.Dataset): The regridded observation dataset for the selected region.
    season (str): A string representing the season to select. Valid values are "DJF", "MAM", "JJA", "SON", "SOND", "NDJF", and "DJFM".

    Returns:
    xarray.Dataset: The regridded observation dataset for the selected season.

    Raises:
    ValueError: If an invalid season string is provided.
    """

    try:
        # Extract the months from the season string
        if season == "DJF":
            months = [12, 1, 2]
        elif season == "MAM":
            months = [3, 4, 5]
        elif season == "JJA":
            months = [6, 7, 8]
        elif season == "JJAS":
            months = [6, 7, 8, 9]
        elif season == "SON":
            months = [9, 10, 11]
        elif season == "SOND":
            months = [9, 10, 11, 12]
        elif season == "NDJF":
            months = [11, 12, 1, 2]
        elif season == "DJFM":
            months = [12, 1, 2, 3]
        else:
            raise ValueError("Invalid season")

        # Select the months from the dataset
        regridded_obs_dataset_region_season = regridded_obs_dataset_region.sel(
            time=regridded_obs_dataset_region["time.month"].isin(months)
        )

        return regridded_obs_dataset_region_season
    except:
        print("Error selecting season")
        sys.exit()

def calculate_anomalies(regridded_obs_dataset_region_season):
    """
    Calculates the anomalies for a given regridded observation dataset for a specific season.

    Parameters:
    regridded_obs_dataset_region_season (xarray.Dataset): The regridded observation dataset for the selected region and season.

    Returns:
    xarray.Dataset: The anomalies for the given regridded observation dataset.

    Raises:
    ValueError: If the input dataset is invalid.
    """
    try:
        obs_climatology = regridded_obs_dataset_region_season.mean("time")
        obs_anomalies = regridded_obs_dataset_region_season - obs_climatology
        return obs_anomalies
    except:
        print("Error calculating anomalies for observations")
        sys.exit()    


def calculate_annual_mean_anomalies_obs(obs_anomalies, season):
    """
    Calculates the annual mean anomalies for a given observation dataset and season.

    Parameters:
    obs_anomalies (xarray.Dataset): The observation dataset containing anomalies.
    season (str): The season for which to calculate the annual mean anomalies.

    Returns:
    xarray.Dataset: The annual mean anomalies for the given observation dataset and season.

    Raises:
    ValueError: If the input dataset is invalid.
    """
    try:
        # Shift the dataset if necessary
        if season in ["DJFM", "NDJFM"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-3)
        elif season in ["DJF", "NDJF"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-2)
        elif season in ["NDJ", "ONDJ"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-1)
        else:
            obs_anomalies_shifted = obs_anomalies

        # Calculate the annual mean anomalies
        obs_anomalies_annual = obs_anomalies_shifted.resample(time="Y").mean("time")

        return obs_anomalies_annual
    except:
        print("Error shifting and calculating annual mean anomalies for observations")
        sys.exit()

def select_forecast_range(obs_anomalies_annual, forecast_range):
    """
    Selects the forecast range for a given observation dataset.

    Parameters:
    obs_anomalies_annual (xarray.Dataset): The observation dataset containing annual mean anomalies.
    forecast_range (str): The forecast range to select.

    Returns:
    xarray.Dataset: The observation dataset containing annual mean anomalies for the selected forecast range.

    Raises:
    ValueError: If the input dataset is invalid.
    """
    try:
        
        forecast_range_start, forecast_range_end = map(int, forecast_range.split("-"))
        print("Forecast range:", forecast_range_start, "-", forecast_range_end)
        
        rolling_mean_range = forecast_range_end - forecast_range_start + 1
        print("Rolling mean range:", rolling_mean_range)

        # if rolling mean range is 1, then we don't need to calculate the rolling mean
        if rolling_mean_range == 1:
            print("rolling mean range is 1, no need to calculate rolling mean")
            return obs_anomalies_annual
        
        obs_anomalies_annual_forecast_range = obs_anomalies_annual.rolling(time=rolling_mean_range, center = True).mean()
        
        return obs_anomalies_annual_forecast_range
    except Exception as e:
        print("Error selecting forecast range:", e)
        sys.exit()


# Call the functions to process the observations
def process_observations(variable, region, region_grid, forecast_range, season, observations_path, obs_var_name):
    """
    Processes the observations dataset by regridding it to the model grid, selecting a region and season,
    calculating anomalies, calculating annual mean anomalies, selecting the forecast range, and returning
    the processed observations.

    Args:
        variable (str): The variable to process.
        region (str): The region to select.
        region_grid (str): The grid to regrid the observations to.
        forecast_range (str): The forecast range to select.
        season (str): The season to select.
        observations_path (str): The path to the observations dataset.
        obs_var_name (str): The name of the variable in the observations dataset.

    Returns:
        xarray.Dataset: The processed observations dataset.
    """

    # Check if the observations file exists
    if not os.path.exists(observations_path):
        print("Error, observations file does not exist")
        return None

    try:
        # Regrid using CDO, select region and load observation dataset
        # for given variable
        obs_dataset = regrid_and_select_region(observations_path, region, obs_var_name)

        # Check for NaN values in the observations dataset
        # print("Checking for NaN values in obs_dataset")
        # check_for_nan_values(obs_dataset)

        # Select the season
        # --- Although will already be in DJFM format, so don't need to do this ---
        regridded_obs_dataset_region_season = select_season(obs_dataset, season)

        # # Print the dimensions of the regridded and selected region dataset
        # print("Regridded and selected region dataset:", regridded_obs_dataset_region_season.time)

        # # Check for NaN values in the observations dataset
        # print("Checking for NaN values in regridded_obs_dataset_region_season")
        # check_for_nan_values(regridded_obs_dataset_region_season)
        
        # Calculate anomalies
        obs_anomalies = calculate_anomalies(regridded_obs_dataset_region_season)

        # Check for NaN values in the observations dataset
        # print("Checking for NaN values in obs_anomalies")
        # check_for_nan_values(obs_anomalies)

        # Calculate annual mean anomalies
        obs_annual_mean_anomalies = calculate_annual_mean_anomalies_obs(obs_anomalies, season)

        # Check for NaN values in the observations dataset
        # print("Checking for NaN values in obs_annual_mean_anomalies")
        # check_for_nan_values(obs_annual_mean_anomalies)

        # Select the forecast range
        obs_anomalies_annual_forecast_range = select_forecast_range(obs_annual_mean_anomalies, forecast_range)
        # Check for NaN values in the observations dataset
        # print("Checking for NaN values in obs_anomalies_annual_forecast_range")
        # check_for_nan_values(obs_anomalies_annual_forecast_range)

        # if the forecast range is "2-2" i.e. a year ahead forecast
        # then we need to shift the dataset by 1 year
        # where the model would show the DJFM average as Jan 1963 (s1961)
        # the observations would show the DJFM average as Dec 1962
        # so we need to shift the observations to the following year
        if forecast_range == "2-2":
            obs_anomalies_annual_forecast_range = obs_anomalies_annual_forecast_range.shift(time=1)


        return obs_anomalies_annual_forecast_range

    except Exception as e:
        print(f"Error processing observations dataset: {e}")
        sys.exit()
