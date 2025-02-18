
# Imports
#from init_sensor_make_model import init_sensor_snowpack, init_sensor_icecolumn
from data_preparation import combine_nc_files,create_input_dataframe, remove_outliers
from data_visualization import plot_measurements
#from translator import CICE_to_SMRT

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

# Remove outliers dataframe
filtered_df = remove_outliers(input_df)
print('outliers removed from input_df...')

# Extract and define relevant variabels 
thickness_ice = filtered_df['hi'] # in m
thickness_snow = filtered_df['hs'] # in m
temperature_profile_ice = filtered_df['temperature_profiles'] # in K
temperature_snow = filtered_df['tsnz'] # in K
salinity_profile_ice = filtered_df['salinity_profiles'] # in kg/kg
#%%

print(temperature_profile_ice.

#%% Translate variables from CICE to SMRT
from SMRT import SMRT_create_ice_columns, SMRT_create_snowpacks,SMRT_create_mediums,SMRT_create_sensor,SMRT_calculate_brightness_temperature
from smrt import make_ice_column, make_snowpack, make_model, sensor_list

# pick how much data to run
n = 100
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


#%%

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

# Access the 1.4 GHz channel using indexing:

#%% create the sensor

for i in range(n):
    sensor =  sensor_list.passive(1.4e9,55)
    n_max_stream = 128 #TB calculation is more accurate if number of streams is increased (currently: default = 32);
    #needs to be increased when using > 1 snow layer on top of sea ice! 
    m = make_model("iba", "dort", rtsolver_options=dict(error_handling='nan', 
                                                        diagonalization_method='shur_forcedtriu', 
                                                        phase_normalization=True, 
                                                        n_max_stream=128))
    
    # run the model for bare sea ice:
    #res1 = m.run(CIMR, ice_columns[i])
    # run the model for snow-covered sea ice:
    res2 = m.run(sensor, mediums[i]) 
    print(res2.TbH(), res2.TbV())
    
    # df = {
    #      'Channel': ['1.4GHz', '6.9GHz', '10.65GHz', '18.7GHz', '37GHz'],
    #      'TB(V)': res2.TbV(),  # Brightness Temperature for Vertical polarization
    #      'TB(H)': res2.TbH(),
    #      'PR':res2.polarization_ratio()  # Polarization Ratio
    #  }
    
    # df = pd.DataFrame(df)
    # # print TBs at horizontal and vertical polarization:
    # #print(res1.TbH(), res1.TbV())
    #print(df)   



#%% 

# Plot SST
plot_measurements(lat=input_df['TLAT'], 
                  lon=input_df['TLON'],
                  colorbar_min = 240,
                  colorbar_max = 275,
                  cvalue=input_df['tair'])




