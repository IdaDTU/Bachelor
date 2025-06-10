from physics import temperature_gradient_snow, calculate_snow_salinity
from SMRT import SMRT_create_ice_columns, SMRT_create_snowpacks,SMRT_create_mediums, SMRT_compute_tbh,SMRT_compute_tbv, scale_brightness_temperature
from dictionaries import OW_tiepoints
import pandas as pd
import numpy as np

# -------------------------- User Input -------------------------- # 

# Create ice layers and snow layers
layers_ice = 1 # select amount of ice layers
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
CICE_directory = 'C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/CICEv2'

# Insert directory for OW data
OW_directory ='C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/OWv2'

# ---------------------------------------------------------------- # 


#%% Read CICE csv
CICE_csv = pd.read_csv(CICE_directory)

# Extract and define relevant scalar variabels 
thickness_ice = CICE_csv['hi'] # in m
thickness_snow = CICE_csv['hs'] # in m
temperature_air = CICE_csv['tair'] # in m
concentration_ice = CICE_csv['aice'] # fraction
salinity_ice = CICE_csv['salinity_average']
temperature_ice = CICE_csv['temperature_average']

# Define geographic location
latitude = CICE_csv['TLAT']
longitude = CICE_csv['TLON']

# Compute snow temperature and density profiles within chosen range
temperature_profile_snow = temperature_gradient_snow(thickness_snow,
                                                     thickness_ice,
                                                     temperature_air,
                                                     layers_snow)

# Constant density profile
denisty_profile_snow = np.array([310,290,270])
denisty_profile_snow = pd.Series([denisty_profile_snow.copy() for i in range(len(temperature_profile_snow))])

#denisty_profile_snow = calculate_snow_density(temperature_profile_snow, 
                                                  # thickness_snow,
                                                  # layers_snow)

snow_salinity = calculate_snow_salinity(thickness_snow, layers_snow, ice_type)

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

#%% Slice inputs
thickness_ice = thickness_ice[start:end]
temperature_ice = temperature_ice[start:end]
salinity_ice = salinity_ice[start:end]
thickness_snow = thickness_snow[start:end]
temperature_profile_snow = temperature_profile_snow[start:end]
snow_salinity = snow_salinity[start:end]  
#denisty_profile_snow = denisty_profile_snow[start:end]
latitude = latitude[start:end]
longitude = longitude[start:end]  


#%% Test

# Defi
#%%n = end - start 
n = 50
# Create ice columns and snowpacks
ice_columns = SMRT_create_ice_columns(thickness_ice,
                                      temperature_ice,
                                      salinity_ice,
                                      n,
                                      ice_type,
                                      layers_ice)
print('finished processing ice columns...')

snowpacks = SMRT_create_snowpacks(thickness_snow,
                                  temperature_profile_snow,
                                  denisty_profile_snow,
                                  snow_salinity,
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
    tb = SMRT_compute_tbh(mediums, frequency, incidence_angle).values
elif polarization == 'v':
    tb = SMRT_compute_tbv(mediums, frequency, incidence_angle).values
else:
    raise ValueError("Polarization must be either 'h' or 'v'.")

# Linearly scale with SIC
# tb_scaled = scale_brightness_temperature(tb ,concentration_ice[:50], OW_tb)
# print('Scaled brigtness temperature computed...')

# Create output DataFrame
tb_df = pd.DataFrame({'tb': tb,
                      'lat': latitude,
                      'lon': longitude})
print('Output dataframe created...')


# Save result as CSVs
tb_df.to_csv(f'C:/Users/user/OneDrive/Desktop/Bachelor/csv/SMRT_{incidence_angle}_{frequency}_{ice_type}_{polarization}_{batch}.csv',index=False)
print('CSV for simulated tb saved...')

# Save result as CSV
tb_df.to_csv(f'/zhome/da/d/187040/bachelor/csv/SMRT/SMRT_{incidence_angle}_{frequency}_{ice_type}_{polarization}_{batch}.csv',index=False)
print('CSV saved...')
