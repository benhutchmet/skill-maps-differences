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
                # if r < 0:
                #     p = np.nan

                # Append the correlation coefficient and p-value to the arrays
                rfield[y, x], pfield[y, x] = r, p

        # Print the range of the correlation coefficients and p-values
        # to 3 decimal places
        print(f"Correlation coefficients range from {rfield.min():.3f} to {rfield.max():.3f}")
        print(f"P-values range from {pfield.min():.3f} to {pfield.max():.3f}")

        # print the correlation coefficients and p-values
        # print("correlation coefficients", rfield)
        print("p-values", pfield)
        # print shape of pfield
        print("shape of pfield", np.shape(pfield))

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

    # Calculate the p-values for the differences between the initialized and uninitialized data
    # Set up an empty array for the t-statistic
    t_stat = np.empty([len(obs_lat), len(obs_lon)])
    p_values = np.empty([len(obs_lat), len(obs_lon)])

    # Loop over the latitudes and longitudes
    for y in range(len(obs_lat)):
        for x in range(len(obs_lon)):
            # set up the model data
            mod_init = init_model_data[:, y, x]
            mod_uninit = uninit_model_data[:, y, x]

            # Calculate the correlation coefficient and p-value
            t_stat[y, x], p_values[y, x] = stats.ttest_ind(mod_init, mod_uninit, equal_var=False)

    # Print the range of the correlation coefficients and p-values
    # to 3 decimal places
    print(f"t-statistic range from {t_stat.min():.3f} to {t_stat.max():.3f}")
    print(f"P-values range from {p_values.min():.3f} to {p_values.max():.3f}")

    # Identify the regions where the correlation improvements are statistically significant
    # at the 95% confidence level
    sign_regions = p_values < p_sig

    # Prin the types of the rfield_diff and sign_regions arrays
    print("rfield_diff type", type(rfield_diff))
    print("sign_regions type", type(sign_regions))

    # Print the shapes of the rfield_diff and sign_regions arrays
    print("rfield_diff shape", np.shape(rfield_diff))
    print("sign_regions shape", np.shape(sign_regions))

    # Return the differences in spatial correlations and the p-values
    return rfield_diff, sign_regions


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
    dcpp_ensemble_mean, dcpp_lat, dcpp_lon, dcpp_years, dcpp_ensemble_members_count = process_model_data_for_plot(dcpp_model_data, dcpp_models)

    # Then process the historical model data to get the ensemble mean
    historical_ensemble_mean, historical_lat, historical_lon, historical_years, historical_ensemble_members_count = process_model_data_for_plot(historical_model_data, historical_models)

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
    # FIXME: lon is not defined here
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
