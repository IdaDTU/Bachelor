
# Imports
#from init_sensor_make_model import init_sensor_snowpack, init_sensor_icecolumn
from data_preparation import combine_nc_files, create_combined_dataframe, remove_nonphysical, SIC_filter
from physics import temperature_gradient_snow, calculate_snow_salinity
import numpy as np
import matplotlib.pyplot as plt
from SMRT import SMRT_create_ice_columns, SMRT_create_snowpacks, SMRT_create_mediums, SMRT_create_sensor, SMRT_compute_tbv, SMRT_compute_tbh
import os
import pandas as pd


#%% input parameters
incidence_angle = 55 # CIMR is 55 deg and MWI is 53.1 deg
frequency =  36.5e9 #(in Hz)
polarization = 'h'
ice_type = 'multiyear'
layers_ice = 1 # select amount of ice layers
layers_snow = 5 # select amount of snow layers

#%%
# Insert .nc data directory
directory = '/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Bachelorprojekt/DMI_data/ny/2024040800/'

# Combine all .nc files in directory into one
combined_nc=combine_nc_files(directory)  
print('combined_nc created...')

# Choose time step
t = 0

# Create subset of dataset
ds_subset = combined_nc.isel(time=t)
print('subset created...')

# Create dataframes
input_df =  create_combined_dataframe(ds_subset, layers_ice)
print('input_df created...')

#%%
# Remove unphysical data from dataframe
filtered_df, OW_df = remove_nonphysical(input_df, layers_ice)
print(f'outliers removed from input_df: {len(input_df)-len(filtered_df)} ...')

# Select SIC range
min_SIC =1
max_SIC =1

# Create dataframe with SIC within chosen range
ice_df = SIC_filter(min_SIC, max_SIC, df=filtered_df)

ice_df = ice_df[:50]

# Extract and define relevant scalar variabels 
thickness_ice = ice_df['hi'] # in m
thickness_snow = ice_df['hs'] # in m
temperature_air = ice_df['tair'] # in m
latitude = ice_df['TLAT']
longitude = ice_df['TLON']
#%%

concentration_ice = ice_df['aice'] # fraction
salinity_ice = ice_df['salinity_average']
temperature_ice = ice_df['temperature_average']

# Compute snow temperature and density profiles within chosen range
temperature_profile_snow = temperature_gradient_snow(thickness_snow,
                                                     thickness_ice,
                                                     temperature_air,
                                                     layers_snow)

# Constant density profile
denisty_profile_snow = np.linspace(310, 270, layers_snow)
denisty_profile_snow = pd.Series([denisty_profile_snow.copy() for i in range(len(temperature_profile_snow))])

#denisty_profile_snow = calculate_snow_density(temperature_profile_snow, 
 #                                                 thickness_snow,
  #                                                layers_snow)
snow_salinity = calculate_snow_salinity(thickness_snow, layers_snow, ice_type)
print('profiles calculated ...')

#%%
n = len(ice_df)

#%%

# Select ice type
ice_type = 'multiyear'

# Create ice_columns and snowpacks
ice_columns = SMRT_create_ice_columns(thickness_ice,
                                      temperature_ice,
                                      salinity_ice,
                                      n,
                                      ice_type,
                                      layers_ice)
print('finished ice columns ...')

#%%

snowpacks = SMRT_create_snowpacks(thickness_snow,
                                  temperature_profile_snow,
                                  denisty_profile_snow,
                                  snow_salinity,
                                  n,
                                  layers_snow)
print('finished snowpacks ...')

# Combine into mediums
mediums = SMRT_create_mediums(snowpacks,
                              ice_columns,
                              n)
print('finished mediums ...')

#%%
# Compute brightness temperatures based on polarization and flatten
if polarization == 'h':
    tb = SMRT_compute_tbh(mediums, frequency, incidence_angle).values
elif polarization == 'v':
    tb = SMRT_compute_tbv(mediums, frequency, incidence_angle).values
else:
    raise ValueError("Polarization must be either 'h' or 'v'.")
    
tb_df = pd.DataFrame({'tb': tb,
                      'lat': latitude,
                      'lon': longitude})
print('Output dataframe created...')



#%%

# Save the DataFrame to a CSV file
tb_df.to_csv(f'/zhome/57/6/168999/Desktop/Bachelor/sensitivity/36.5GHz/sensitivity_MY_H_i{layers_ice}_s{layers_snow}.csv', index=False)

