
# Imports
from init_sensor_make_model import init_sensor_snowpack, init_sensor_icecolumn
from create_input_df import combine_nc_files,create_input_dataframe
from thermal_gradients import compute_thermal_gradients
from measurement_visulazations import plot_measurements

# Insert .nc data directory
directory = 'C:/Users/user/OneDrive/Desktop/Bachelor/Arctic_data' 

# Combine all .nc files in directory into one
combined_nc=combine_nc_files(directory)  

# Create subset of dataset
ds_subset = combined_nc.isel(time=23,            # select time step
                             nj=slice(0, 1490),  # 1490 is the max lenght of the nj dimension
                             ni=slice(0, 1115))  # 1115 is the max lenght of the ni dimension

# Create dataframe
input_df = create_input_dataframe(ds_subset)

# Extract and define relevant variabels 
thickness_ice = input_df['hi']                   # in m
thickness_snow = input_df['hs']                  # in m
temperature_air = input_df['tair']               # in K
temperature_water = input_df['sst']              # in K
temperature_ice = input_df['tinz']               # in K

#%%
print(temperature_air.describe())


#%% Plot SST
plot_measurements(lat=input_df['TLAT'], 
                  lon=input_df['TLON'],
                  colorbar_min = 240,
                  colorbar_max = 275,
                  cvalue=input_df['tair'])


#%% SMRT outputs
MWI_output_df_snowpack = init_sensor_snowpack(sensor='MWI',
                                              thickness=[100],
                                              corr_length=[5e-5],
                                              microstructure_model='exponential',
                                              density=320)


CIMR_output_df_snowpack = init_sensor_snowpack(sensor='CIMR',
                                               thickness=[100],
                                               corr_length=[5e-5],
                                               microstructure_model='exponential',
                                               density=320)

#print(CIMR_df)
#print(MWI_df)


#%% Loop over each row in the DataFrame
import numpy as np


# Store all temperature profiles
temperature_profiles = []

# Compute temperature profile for each point
for i in range(len(input_df)):
    depths = np.linspace(0, thickness_ice.iloc[i] + thickness_snow.iloc[i], n) 
    temperatures = compute_thermal_gradients(
        temperature_air.iloc[i], 
        temperature_water.iloc[i], 
        thickness_ice.iloc[i], 
        thickness_snow.iloc[i], 
        depths) 
   
    temperature_profiles.append(temperatures)  # Store the list


corr_length = 1.0e-3
salinity = 7.88-1.59 * thickness_ice
porosity = salinity * (0.05322 - 4.919/temperature_ice) # ice porosity in fractions, [0..1]

#%% Storage for results



test = init_sensor_icecolumn(sensor='CIMR',
                          ice_type='multiyear',
                          thickness_ice= [thickness_ice.iloc[1]],
                          temperature=temperature_profiles[1],
                          microstructure_model='exponential',
                          corr_length=corr_length,
                          brine_inclusion_shape='spheres',
                          salinity=salinity.iloc[1],
                          porosity=porosity.iloc[1],
                          density = 2,
                          add_water_substrate='ocean')
print(test)



#%%
TBice = []

for i in range(len(input_df)):
    result = init_sensor_icecolumn(sensor='CIMR',
                          ice_type='firstyear',
                          thickness_ice=[thickness_ice.iloc[i]],
                          temperature=temperature_profiles[i],
                          microstructure_model='exponential',
                          corr_length=1.0e-3,
                          brine_inclusion_shape='spheres',
                          salinity=(salinity).iloc[i],      #from permille to ppt
                          brine_volume_fraction=0.0,   #page 101
                          brine_volume_model=None,
                          brine_permittivity_model=None,
                          ice_permittivity_model=None, 
                          saline_ice_permittivity_model=None,
                          porosity=porosity.iloc[i], 
                          density=None, 
                          add_water_substrate=True,
                          surface=None, 
                          interface=None,
                          substrate=None,
                          atmosphere=None
                          )
    TBice.append(result)
    
print(TBice)








