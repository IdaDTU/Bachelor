
# Imports
#from init_sensor_make_model import init_sensor_snowpack, init_sensor_icecolumn
from data_preparation import combine_nc_files, create_combined_dataframe, remove_nonphysical, SIC_filter
from physics import temperature_gradient_snow, calculate_snow_density
from sensitivity import sensitivity_snowpacks,sensitivity_ice_columns
import numpy as np
import matplotlib.pyplot as plt
from SMRT import SMRT_create_ice_columns, SMRT_create_snowpacks, SMRT_create_mediums, SMRT_create_sensor, SMRT_compute_tbv

# Insert .nc data directory
directory = 'C:/Users/user/OneDrive/Desktop/Bachelor/cold_data_new/2024041200' # Ida directory 
#
# Combine all .nc files in directory into one
combined_nc=combine_nc_files(directory)  
print('combined_nc created...')

# Choose time step
t = 0

# Create subset of dataset
ds_subset = combined_nc.isel(time=t)
print('subset created...')

# Create ice layers and snow layers
layers_ice = 4 # select amount of ice layers
layers_snow = 1 # select amount of snow layers

# Create dataframes
input_df =  create_combined_dataframe(ds_subset, layers_ice)
print('input_df created...')

# Remove unphysical data from dataframe
filtered_df = remove_nonphysical(input_df, layers_ice)
print(f'outliers removed from input_df: {len(input_df)-len(filtered_df)} ...')

# Select SIC range
min_SIC =1
max_SIC =1

# Create dataframe with SIC within chosen range
ice_df = SIC_filter(min_SIC, max_SIC, df=filtered_df)

# Extract and define relevant scalar variabels 
thickness_ice = ice_df['hi'] # in m
thickness_snow = ice_df['hs'] # in m
temperature_air = ice_df['tair'] # in m

# Extract ice profiles  within chosen range
salinity_profile_ice = ice_df['salinity_profiles'] # in kg/kg
temperature_profile_ice = ice_df['temperature_profiles'] # in K


# Compute snow temperature and density profiles within chosen range
temperature_profile_snow = temperature_gradient_snow(thickness_snow,
                                                     thickness_ice,
                                                     temperature_air,
                                                     layers_snow)

denisty_profile_snow = calculate_snow_density(temperature_profile_snow, 
                                                  thickness_snow,
                                                  layers_snow)
print('profiles calculated ...')

#%%
n = len(ice_df)

#%%

# Select ice type
ice_type = 'multiyear'

# Create ice_columns and snowpacks
ice_columns = SMRT_create_ice_columns(thickness_ice,
                                      temperature_profile_ice,
                                      salinity_profile_ice,
                                      n,
                                      ice_type,
                                      layers_ice)
print('finished ice columns ...')

snowpacks = SMRT_create_snowpacks(thickness_snow,
                                  temperature_profile_snow,
                                  denisty_profile_snow,
                                  n,
                                  layers_snow)
print('finished snowpacks ...')

# Combine into mediums
mediums = SMRT_create_mediums(snowpacks,
                              ice_columns,
                              n)
print('finished mediums ...')

incidence_angle = 55 # CIMR is 55 deg and MWI is 53.1 deg
frequency =  36.5e9 #(in Hz)
#%%
SMRT_df = SMRT_compute_tbv(mediums, frequency, incidence_angle)





#%%
print(SMRT_df.describe())


# Save the DataFrame to a CSV file
SMRT_df.to_csv(f'C:/Users/user/OneDrive/Desktop/Bachelor/csv/sensitivity_MY_i{layers_ice}_s{layers_snow}.csv', index=False)




