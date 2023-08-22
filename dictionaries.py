# python dictionaries to be used in python/functions.py

# define the base directory where the data is stored
base_dir = "/home/users/benhutch/skill-maps-processed-data"

# define the directory where the plots will be saved
plots_dir = base_dir + "/plots"

gif_plots_dir = base_dir + "/plots/gif"

# list of the test model
test_model = [ "CMCC-CM2-SR5" ]
test_model_bcc = [ "BCC-CSM2-MR" ]
test_model2 = [ "EC-Earth3" ]
test_model_norcpm = [ "NorCPM1" ]
test_model_hadgem = [ "HadGEM3-GC31-MM" ]
test_model_cesm = [ "CESM1-1-CAM5-CMIP5" ]

# List of the full models
models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5", "NorCPM1" ]

# define the paths for the observations
obs_psl = "/home/users/benhutch/ERA5_psl/long-ERA5-full.nc"

# For the north atlantic region
obs_psl_na = "/home/users/benhutch/ERA5_psl/long-ERA5-full-north-atlantic.nc"

# Global
obs_psl_glob = "/home/users/benhutch/ERA5_psl/long-ERA5-full-global.nc"

# the variable has to be extracted from these
obs_tas = "/home/users/benhutch/ERA5/adaptor.mars.internal-1687448519.6842003-11056-8-3ea80a0a-4964-4995-bc42-7510a92e907b.nc"
obs_sfcWind = "/home/users/benhutch/ERA5/adaptor.mars.internal-1687448519.6842003-11056-8-3ea80a0a-4964-4995-bc42-7510a92e907b.nc"
#

# Set up the base directory where the historical data is stored
base_dir_historical = "/home/users/benhutch/skill-maps-processed-data/historical"

# set up the base path example for the historical runs
base_path_example = "/badc/cmip6/data/CMIP6/CMIP"

gridspec_path = "/home/users/benhutch/gridspec"

# set up the base path for the home dir
home_dir = "/home/users/benhutch"

canari_base_path_historical = "/gws/nopw/j04/canari/users/benhutch/historical"

obs_rsds="not/yet/implemented"

obs = "/home/users/benhutch/ERA5/adaptor.mars.internal-1691509121.3261805-29348-4-3a487c76-fc7b-421f-b5be-7436e2eb78d7.nc"

gridspec_global = "/home/users/benhutch/gridspec/gridspec-global.txt"

gridspec_north_atlantic = "/home/users/benhutch/gridspec/gridspec-north-atlantic.txt"

obs_regrid = "/home/users/benhutch/ERA5/ERA5_full_global.nc"

# Define the labels for the plots - wind
sfc_wind_label="10-metre wind speed"
sfc_wind_units = 'm s\u207b\u00b9'

# Define the labels for the plots - temperature
tas_label="2-metre temperature"
tas_units="K"

psl_label="Sea level pressure"
psl_units="hPa"

rsds_label="Surface solar radiation downwards"
rsds_units="W m\u207b\u00b2"

# Define the dimensions for the grids
# for processing the observations
north_atlantic_grid = {
    'lon1': 280,
    'lon2': 37.5,
    'lat1': 77.5,
    'lat2': 20
}

# define the NA grid for obs
north_atlantic_grid_obs = {
    'lon1': 100,
    'lon2': 217.5,
    'lat1': 77.5,
    'lat2': 20
}

# Define the dimensions for the gridbox for the azores
azores_grid = {
    'lon1': 152,
    'lon2': 160,
    'lat1': 36,
    'lat2': 40
}

# Define the dimensions for the gridbox for iceland
iceland_grid = {
    'lon1': 155,
    'lon2': 164,
    'lat1': 63,
    'lat2': 70
}

# Define the dimensions for the gridbox for the N-S UK index
# From thornton et al. 2019
uk_n_box = {
    'lon1': 153,
    'lon2': 201,
    'lat1': 57,
    'lat2': 70
}

# And for the southern box
uk_s_box = {
    'lon1': 153,
    'lon2': 201,
    'lat1': 38,
    'lat2': 51
}

# models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5", "NorCPM1" ]

# Set up the model_dictionary for the sfcWind historical models
model_dictionary_sfcWind_historical_badc = [
    {'model_name': 'BCC-CSM2-MR', 'runs': '1-3', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'MPI-ESM1-2-HR', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'CanESM5', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'CMCC-CM2-SR5', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'HadGEM3-GC31-MM', 'runs': '1-4', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '3'},
    {'model_name': 'EC-Earth3', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'MPI-ESM1-2-LR', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'FGOALS-f3-L', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'MIROC6', 'runs': '1-10', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'IPSL-CM6A-LR', 'runs': '1-31', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'}
]

# set up the numbers for the sfcWind historical models
# 10 models
model_dictionary_sfcWind_historical_badc_numbers = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]

# Define the sfcWind models list
sfcWind_models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR" ]

#models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5", "NorCPM1" ]
# Set up the model_dictionary for the psl historical models
# stored in the CMIP6 archive on JASMIN (badc)
model_dictionary_psl_historical_badc = [
    {'model_name': 'BCC-CSM2-MR', 'runs': '1-3', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'MPI-ESM1-2-HR', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'CanESM5', 'runs': '1-25', 'init_schemes': '1', 'physics_scheme': '1,2', 'forcing_scheme': '1'},
    {'model_name': 'CMCC-CM2-SR5', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'HadGEM3-GC31-MM', 'runs': '1-4', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '3'},
    {'model_name': 'EC-Earth3', 'runs': '101-150', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'MPI-ESM1-2-LR', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'FGOALS-f3-L', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'MIROC6', 'runs': '1-50', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'IPSL-CM6A-LR', 'runs': '1-31', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'NorCPM1', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'}
]

# set up the numbers for the psl historical models
model_dictionary_psl_historical_badc_numbers = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12 ]

# Define the psl models list
# 10 models
psl_models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "NorCPM1" ]

# models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5", "NorCPM1" ]
# Set up the model_dictionary for the tas historical models
# stored in the CMIP6 archive on JASMIN (badc)
model_dictionary_tas_historical_badc = [
    {'model_name': 'BCC-CSM2-MR', 'runs': '1-3', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'MPI-ESM1-2-HR', 'runs': '1-10', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'CanESM5', 'runs': '1-40', 'init_schemes': '1', 'physics_scheme': '1,2', 'forcing_scheme': '1'},
    {'model_name': 'CMCC-CM2-SR5', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'HadGEM3-GC31-MM', 'runs': '1-4', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '3'},
    {'model_name': 'EC-Earth3', 'runs': '101-150', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'MPI-ESM1-2-LR', 'runs': '1-30', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'FGOALS-f3-L', 'runs': '1-3', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'MIROC6', 'runs': '1-50', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'IPSL-CM6A-LR', 'runs': '1-32', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'NorCPM1', 'runs': '30', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'}
]

# set up the numbers for the tas historical models
model_dictionary_tas_historical_badc_numbers = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12 ]

# Define the tas models list
# 11 models
tas_models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "NorCPM1" ]

# models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5", "NorCPM1" ]
# Set up the model_dictionary for the rsds historical models
# stored in the CMIP6 archive on JASMIN (badc)
model_dictionary_rsds_historical_badc = [
    {'model_name': 'BCC-CSM2-MR', 'runs': '1-3', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'MPI-ESM1-2-HR', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'CanESM5', 'runs': '1-25', 'init_schemes': '1', 'physics_scheme': '1,2', 'forcing_scheme': '1'},
    {'model_name': 'CMCC-CM2-SR5', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'HadGEM3-GC31-MM', 'runs': '1-4', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '3'},
    {'model_name': 'EC-Earth3', 'runs': '1,2,22', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'MPI-ESM1-2-LR', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'FGOALS-f3-L', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'MIROC6', 'runs': '1-10', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'IPSL-CM6A-LR', 'runs': '1-31', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'},
    {'model_name': 'NorCPM1', 'runs': '1', 'init_schemes': '1', 'physics_scheme': '1', 'forcing_scheme': '1'}
]

# set up the numbers for the rsds historical models
model_dictionary_rsds_historical_badc_numbers = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12 ]

# Set up the rsds models list
# 11 models
rsds_models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "NorCPM1" ]

# Create a list of the model dictionaries
model_dictionary_list = [model_dictionary_sfcWind_historical_badc, model_dictionary_psl_historical_badc, model_dictionary_tas_historical_badc, model_dictionary_rsds_historical_badc]

# TODO - define model dictionary for tos Omon

season_months = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "JJAS": [6, 7, 8, 9],
    "SON": [9, 10, 11],
    "SOND": [9, 10, 11, 12],
    "NDJF": [11, 12, 1, 2],
    "DJFM": [12, 1, 2, 3]
}

season_timeshift = [
    {'season': 'DJF', 'timeshift': -2},
    {'season': 'NDJF', 'timeshift': -2},
    {'season': 'DJFM', 'timeshift': -3},
    {'season': 'NDJFM', 'timeshift': -3},
    {'season': 'NDJ', 'timeshift': -1},
    {'season': 'ONDJ', 'timeshift': -1},
]

variables = [ 'psl', 'tas', 'sfcWind', 'rsds', 'tos' ]
