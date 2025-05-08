from physics import temperature_gradient_snow, calculate_snow_density
from SMRT import SMRT_create_ice_columns, SMRT_create_snowpacks,SMRT_create_mediums, SMRT_compute_tbh,SMRT_compute_tbv, scale_brightness_temperature,set_OW_tb
from dictionaries import OW_tiepoints
import pandas as pd
import numpy as np

# -------------------------- User Input -------------------------- # 

# Create ice layers and snow layers
layers_ice = 2 # select amount of ice layers
layers_snow = 3 # select amount of snow layers

# Select freqensies
incidence_angle = 55 # CIMR is 55 deg and MWI is 53.1 deg
frequency =  18.7e9 #(in Hz)

# Desired batch index (starting from 1)
batch = 1  # Change this to 1-10

# OW tiepoint
OW_tb = OW_tiepoints["36.5H"]

# Choose type of ice to simulate
ice_type = 'multiyear'

# Choose polarization to compute
polarization = 'h' # choose h or v

# Insert directory for CICE data for sea ice
CICE_directory = 'C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/CICE'

# Insert directory for OW data
OW_directory ='C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/OW'

# ---------------------------------------------------------------- # 

#%% Read CICE csv

#%% Read CICE csv
CICE_csv = pd.read_csv(CICE_directory)

# Extract and define relevant scalar variabels 
thickness_ice = CICE_csv['hi'] # in m
thickness_snow = CICE_csv['hs'] # in m
temperature_air = CICE_csv['tair'] # in m
accumulation_rate = CICE_csv['snow'] #in cm/day
concentration_ice = CICE_csv['aice'] # fraction

# Define geographic location
latitude = CICE_csv['TLAT']
longitude = CICE_csv['TLON']

# Extract ice profiles  within chosen range and convert stringified arrays to actual NumPy arrays
salinity_profile_ice = CICE_csv['salinity_profiles'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))  # in kg/kg
temperature_profile_ice = CICE_csv['temperature_profiles'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))  # in K

# Compute snow temperature and density profiles within chosen range
temperature_profile_snow = temperature_gradient_snow(thickness_snow,
                                                     thickness_ice,
                                                     temperature_air,
                                                     layers_snow)

denisty_profile_snow = calculate_snow_density(temperature_profile_snow, 
                                                  thickness_snow,
                                                  layers_snow)

# Split total samples into 10 batches as evenly as possible
total_samples =len(CICE_csv)
base_batch_size = total_samples // 10
remainder = total_samples % 10

# Create the batch_sizes list
batch_sizes = [base_batch_size + 1 if i < remainder else base_batch_size for i in range(10)]

# Compute start and end using cumulative sum
cumulative_sizes = np.cumsum(batch_sizes)

# Get start and end for the given batch
if batch == 1:
    start = 0
else:
    start = cumulative_sizes[batch - 2]
end = cumulative_sizes[batch - 1]
print(f"Processing batch {batch}: start={start}, end={end}")

# Slice inputs
thickness_ice = thickness_ice[start:end]
temperature_profile_ice = temperature_profile_ice[start:end]
salinity_profile_ice = salinity_profile_ice[start:end]
thickness_snow = thickness_snow[start:end]
temperature_profile_snow = temperature_profile_snow[start:end]
denisty_profile_snow = denisty_profile_snow[start:end]
latitude = latitude[start:end]
longitude = longitude[start:end]  

n = end - start 

# Create ice columns and snowpacks
ice_columns = SMRT_create_ice_columns(thickness_ice,
                                      temperature_profile_ice,
                                      salinity_profile_ice,
                                      n,
                                      ice_type,
                                      layers_ice)
print('finished processing ice columns...')

snowpacks = SMRT_create_snowpacks(thickness_snow,
                                  temperature_profile_snow,
                                  denisty_profile_snow,
                                  n,
                                  layers_snow)
print('finished processing snowpacks...')

# Combine into mediums
mediums = SMRT_create_mediums(snowpacks,
                              ice_columns,
                              n)
print('finished processing mediums...')

# Compute brightness temperatures based on polarization and flatten
if polarization == 'h':
    tb = SMRT_compute_tbh(mediums, frequency, incidence_angle)
elif polarization == 'v':
    tb = SMRT_compute_tbv(mediums, frequency, incidence_angle)
else:
    raise ValueError("Polarization must be either 'h' or 'v'.")

# Linearly scale with SIC
tb_scaled = scale_brightness_temperature(tb ,concentration_ice.iloc, OW_tb)
print('Scaled brigtness temperature computed...')

# Create output DataFrame
tb_df = pd.DataFrame({'tb': tb_scaled,
                      'lat': latitude.values,
                      'lon': longitude.values})
print('Output dataframe created...')

# Save result as CSVs
tb_df.to_csv(f'C:/Users/user/OneDrive/Desktop/Bachelor/csv/SMRT_{incidence_angle}_{frequency}_{ice_type}_{polarization}_{batch}.csv',index=False)
print('CSV for simulated tb saved...')

# Save result as CSV
tb_df.to_csv(f'/zhome/da/d/187040/bachelor/csv/SMRT/SMRT_{incidence_angle}_{frequency}_{ice_type}_{polarization}_{batch}.csv',index=False)
print('CSV saved...')
