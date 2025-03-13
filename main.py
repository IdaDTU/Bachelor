
# Imports
#from init_sensor_make_model import init_sensor_snowpack, init_sensor_icecolumn
from data_preparation import combine_nc_files, create_input_dataframe, remove_nonphysical
from data_visualization import plot_measurements
from physics import temperature_gradient_snow, calculate_snow_density, firn_densification
from SMRT import SMRT_create_ice_columns, SMRT_create_snowpacks,SMRT_create_mediums,SMRT_create_sensor, SMRT_compute_tbv
from smrt import make_ice_column, make_snowpack, make_model, sensor_list

# Insert .nc data directory
directory = 'C:/Users/user/OneDrive/Desktop/Bachelor/test' # Ida directory 
# directory = '/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Bachelorprojekt/DMI_data' # Josephine directory

# Combine all .nc files in directory into one
combined_nc=combine_nc_files(directory)  
print('combined_nc created...')

# Create subset of dataset
ds_subset = combined_nc.isel(time = 0)  # select time step
print('subset created...')

# Create ice layers and snow layers
layers_ice = 2 # select amount of ice layers
layers_snow = 3 # select amount of snow layers

# Create dataframe and select amount of ice layers
input_df = create_input_dataframe(ds_subset, layers_ice)
print('input_df created...')

# Remove unphysical data from dataframe
filtered_df = remove_nonphysical(input_df, layers_ice)
print(f'outliers removed from input_df: {len(input_df)-len(filtered_df)} ...')

# Extract and define relevant scalar variabels 
thickness_ice = filtered_df['hi'] # in m
thickness_snow = filtered_df['hs'] # in m
temperature_air = filtered_df['tair'] # in m
print('variables extracted...')

# Extract ice profiles
salinity_profile_ice = filtered_df['salinity_profiles'] # in kg/kg
temperature_profile_ice = filtered_df['temperature_profiles'] # in K

# Calculate snow temperature and density profiles
temperature_profile_snow = temperature_gradient_snow(thickness_snow,
                                                     thickness_ice,
                                                     temperature_air,
                                                     layers_snow)

denisty_profile_snow = calculate_snow_density(thickness_snow,
                                      temperature_profile_snow,
                                      layers_snow)

print('profiles extracted and computed...')

# Pick how much data to run
# n = len(filtered_df)
n = 10

# Choose type of ice to simulate
ice_type = 'multiyear'

# Create ice_columns and snowpacks
ice_columns = SMRT_create_ice_columns(thickness_ice,
                                      temperature_profile_ice,
                                      salinity_profile_ice,
                                      n,
                                      ice_type,
                                      layers_ice)

snowpacks = SMRT_create_snowpacks(thickness_snow,
                                  temperature_profile_snow,
                                  denisty_profile_snow,
                                  n,
                                  layers_snow)

# Combine into mediums
mediums = SMRT_create_mediums(snowpacks,
                              ice_columns,
                              n)

# Compute brightness temperatures
tbH_df = SMRT_compute_tbv(mediums, 37e9, 55.0)

# Plot tb
plot_measurements(lat=filtered_df['TLAT'][:n], 
                  lon=filtered_df['TLON'][:n],
                  cvalue=tbH_df['tbh'])




