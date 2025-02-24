
# Imports
#from init_sensor_make_model import init_sensor_snowpack, init_sensor_icecolumn
from data_preparation import combine_nc_files,create_input_dataframe, remove_nonphysical
from data_visualization import plot_measurements
import pandas as pd

# Insert .nc data directory
directory = 'C:/Users/user/OneDrive/Desktop/Bachelor/test' # Ida directory 
# directory = '/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Bachelorprojekt/DMI_data' # Josephine directory

# Combine all .nc files in directory into one
combined_nc=combine_nc_files(directory)  
print('combined_nc created...')

# Create subset of dataset
ds_subset = combined_nc.isel(time = 0)  # select time step
print('subset created...')

# Create dataframe
input_df = create_input_dataframe(ds_subset)
print('input_df created...')

#%% Remove outliers dataframe
filtered_df = remove_nonphysical(input_df)
print(f'outliers removed from input_df: {len(input_df)-len(filtered_df)} ...')

# Extract and define relevant variabels 
thickness_ice = filtered_df['hi'] # in m
thickness_snow = filtered_df['hs'] # in m
temperature_profile_ice = filtered_df['temperature_profiles'] # in K
temperature_snow = filtered_df['tsnz'] # in K
salinity_profile_ice = filtered_df['salinity_profiles'] # in kg/kg

#%% Calculate porosity using Frankenstein and Garner [1967]

# porosity = salinity_profile_ice[1000]*(0.05322-4.919/temperature_profile_ice[1000])
# print(porosity) # okay de er ret lave....

print(salinity_profile_ice)

print(temperature_profile_ice)

#%% Translate variables from CICE to SMRT
from SMRT import SMRT_create_ice_columns, SMRT_create_snowpacks,SMRT_create_mediums,SMRT_create_sensor#,SMRT_calculate_brightness_temperature
from smrt import make_ice_column, make_snowpack, make_model, sensor_list

# pick how much data to run
n = 10

#%%
ice_columns = SMRT_create_ice_columns(thickness_ice,
                                      temperature_profile_ice,
                                      salinity_profile_ice,
                                      n)

snowpacks = SMRT_create_snowpacks(thickness_snow,
                                  temperature_snow,
                                  n)

mediums = SMRT_create_mediums(snowpacks,
                              ice_columns,
                              n)


 # Define CIMR sensor with specified frequencies, incidence angle, and polarizations
CIMR = sensor_list.passive(frequency = [1.4e9, 6.9e9, 10.65e9, 18.7e9, 36.5e9],  # Frequencies in Hz
                           theta = 55,  # Incidence angle in degrees
                           polarization = ['V', 'H'],  # Vertical and Horizontal polarization
                           channel_map=['1.4', '6.9', '10.65', '18.7', '37'],  # Frequency labels in GHz
                           name ='CIMR')  # Sensor name

# Define MWI sensor with its frequencies, incidence angle, and polarizations
MWI = sensor_list.passive([18.7e9, 23.8e9, 31.4e9, 50.3e9, 52.7e9, 53.24e9, 53.75e9, 89e9], 
                          53.1,  # Incidence angle in degrees
                          ['V', 'H'],  # Vertical and Horizontal polarization
                          [18.7, 23.8, 31.4, 50.3, 52.7, 53.24, 53.75, 89],  # Frequency labels in GHz
                          'MWI')  # Sensor name


#%% create the sensor

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


#%% 
# Plot tb
plot_measurements(lat=filtered_df['TLAT'][:n], 
                  lon=filtered_df['TLON'][:n],
                  colorbar_min = 240,
                  colorbar_max = 275,
                  cvalue=[tbV])




