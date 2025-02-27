
# Imports
#from init_sensor_make_model import init_sensor_snowpack, init_sensor_icecolumn
from data_preparation import combine_nc_files, create_input_dataframe, remove_nonphysical, temperature_gradient_snow
from data_visualization import plot_measurements
import pandas as pd
from SMRT import SMRT_create_ice_columns, SMRT_create_snowpacks,SMRT_create_mediums,SMRT_create_sensor#,SMRT_calculate_brightness_temperature
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

# Selcect amount of ice layers and snow layers
layers_ice = 2
layers_snow = 3

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

# Calculate snow temperature profile
temperature_profile_snow = temperature_gradient_snow(thickness_snow,
                                                     thickness_ice,
                                                     temperature_air,
                                                     layers_snow)
print('profiles extracted and computed...')

# pick how much data to run
# n = len(filtered_df)
n = 10

ice_columns = SMRT_create_ice_columns(thickness_ice,
                                      temperature_profile_ice,
                                      salinity_profile_ice,
                                      n,
                                      layers_ice)
#
snowpacks = SMRT_create_snowpacks(thickness_snow,
                                  temperature_profile_snow,
                                  n,
                                  layers_snow)

mediums = SMRT_create_mediums(snowpacks,
                              ice_columns,
                              n)

results = []

for i, medium in enumerate(mediums):
    # Process only every 500th iteration
    #if i % 400 != 0:
    #    continue

    print(f"Processing index: {i}")
    
    # Create the sensor object for a given frequency and incidence angle.
    sensor = sensor_list.passive(1.4e9, 55.0)
    
    # Increase the number of streams for a more accurate TB calculation
    n_max_stream = 128  # Default is 32; increase if using > 1 snow layer on top of sea ice.
    m = make_model("iba", "dort", rtsolver_options={"n_max_stream": n_max_stream})
    
    # Run the model for snow-covered sea ice:
    res = m.run(sensor, medium)
    
    # Compute the brightness temperature in vertical polarization.
    tbv = res.TbV()
    results.append(tbv)

# Create a DataFrame to store the results.
tbV = pd.DataFrame({'tbv': results})

# Plot tb
plot_measurements(lat=filtered_df['TLAT'][:n], 
                  lon=filtered_df['TLON'][:n],
                  cvalue=[tbV])




