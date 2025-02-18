
# Imports
#from init_sensor_make_model import init_sensor_snowpack, init_sensor_icecolumn
from data_preparation import combine_nc_files,create_input_dataframe
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

# Extract and define relevant variabels 
thickness_ice = input_df['hi'] # in m
thickness_snow = input_df['hs'] # in m
temperature_profile_ice = input_df['temperature_profiles'] # in K
temperature_snow = input_df['tsnz'] # in K
salinity_profile_ice = input_df['salinity_profiles'] # in kg/kg


#%% Translate variables from CICE to SMRT
import numpy as np
from smrt import make_ice_column, make_snowpack, make_model, sensor_list
from smrt import PSU

# Prepare a list to hold each ice column.
ice_columns = []
# Loop over each index to create the ice columns.
for i in range(2):
    l = 7  # 7 ice layers

    # Distribute the total ice thickness evenly across the layers.
    thickness = np.array([thickness_ice.iloc[i] / l] * l)
    
    # Define the correlation length for each layer.
    p_ex = np.array([1.0e-3] * l)
    
    # Extract the temperature profile (a gradient) for the current iteration.
    temperature = temperature_profile_ice.iloc[i]
    
    # Extract the salinity profile and apply the PSU conversion.
    salinity = salinity_profile_ice.iloc[i] 
    
    # Define additional properties for the ice column.
    ice_type = 'multiyear'  # Options: 'first-year' or 'multiyear'
    porosity = 0.08        # Ice porosity (fraction between 0 and 1)
    
    # Create the ice column using the provided parameters.
    ice_column = make_ice_column(
        ice_type=ice_type,
        thickness=thickness,
        temperature=temperature,
        microstructure_model="exponential",
        brine_inclusion_shape="spheres",  # Options: "spheres", "random_needles", or "mix_spheres_needles"
        salinity=salinity,  # If salinity is provided, the model calculates the brine volume fraction.
        porosity=porosity,  # If porosity is provided, the model calculates the density.
        corr_length=p_ex,
        add_water_substrate="ocean"  # Adds a water substrate beneath the ice column.
    )
    
    # Store the created ice column.
    ice_columns.append(ice_column)
#%%
# create snowpack with 1 snow layers:
# Prepare a list to hold each snowpack.
snowpacks = []
for i in range(2):    
    l_s=1 #1 snow layers
    thickness_s = np.array([thickness_snow.iloc[i]])
    p_ex_s = np.array([2e-5]*l_s)
    temperature_s = temperature_snow.iloc[i]
    density_s = [300]
    
    # create the snowpack
    snowpack = make_snowpack(thickness=thickness_s,
                             microstructure_model="exponential",
                             density=density_s,
                             temperature=temperature_s,
                             corr_length=p_ex_s)

    # Store the created snowpack.
    snowpacks.append(snowpack)



#%%   
#add snowpack on top of ice column:
mediums=[]
for i in range(2):
    medium = snowpacks[i] + ice_columns[i]
    mediums.append(medium)
print(mediums)    

#%% create the sensor
for i in range(2):
    sensor = sensor_list.passive(1.6e9, 55.)
    
    n_max_stream = 128 #TB calculation is more accurate if number of streams is increased (currently: default = 32);
    #needs to be increased when using > 1 snow layer on top of sea ice! 
    m = make_model("iba", "dort", rtsolver_options ={"n_max_stream": n_max_stream})
    
    # run the model for bare sea ice:
    res1 = m.run(sensor, ice_columns[i])
    # run the model for snow-covered sea ice:
    res2 = m.run(sensor, mediums[i])
    
    # print TBs at horizontal and vertical polarization:
    print(res1.TbH(), res1.TbV())
    print(res2.TbH(), res2.TbV())
    





#%% 

# Plot SST
plot_measurements(lat=input_df['TLAT'], 
                  lon=input_df['TLON'],
                  colorbar_min = 240,
                  colorbar_max = 275,
                  cvalue=input_df['tair'])




