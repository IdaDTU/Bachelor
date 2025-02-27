import numpy as np
import pandas as pd
from smrt import make_ice_column, make_snowpack, make_model, sensor_list

def SMRT_create_ice_columns(thickness_ice,
                temperature_profile_ice,
                salinity_profile_ice,
                n,
                layers_ice):
    
    ice_columns = []
    # Loop over each index to create the ice columns.
    for i in range(n):
        l = layers_ice  #  ice layers
        print(f"Processing ice_columns index {i}. Finished: {(i/n)}...")
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
        porosity = 0.08       # Ice porosity (fraction between 0 and 1)
        
        # Create the ice column using the provided parameters.
        ice_column = make_ice_column(ice_type=ice_type,
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
        
    return ice_columns

def SMRT_create_snowpacks(thickness_snow,
                          temperature_profile_snow,
                          n,
                          layers_snow):
    snowpacks = []
    for i in range(n):    
        l_s=layers_snow #3 snow layers
        print(f"Processing snowpacks index: {i}. Finished: {(i/n)}...")
        thickness_s = np.linspace(0, thickness_snow.iloc[i], l_s)
        p_ex_s = np.array([1.5e-4]*l_s) 
        temperature_s = temperature_profile_snow.iloc[i]
        density_s = [200,300, 340]
        
        # create the snowpack
        snowpack = make_snowpack(thickness=thickness_s,
                                 microstructure_model="exponential",
                                 density=density_s,
                                 temperature=temperature_s,
                                 corr_length=p_ex_s)
    
        # Store the created snowpack.
        snowpacks.append(snowpack)
    return snowpacks
    
def SMRT_create_mediums(snowpacks,
                       ice_columns,
                       n):
    mediums=[]
    for i in range(n):
        print(f"Processing mediums index: {i}")
        medium = snowpacks[i] + ice_columns[i]
        mediums.append(medium) 
    
    return mediums

def SMRT_create_sensor(name):   
    if name == 'CIMR':
        # Define CIMR sensor with specified frequencies, incidence angle, and polarizations
        CIMR = sensor_list.passive(
            [1.4e9, 6.9e9, 10.65e9, 18.7e9, 36.5e9],  # Frequencies in Hz
            55,                                         # Incidence angle in degrees
            ['V', 'H'],                                 # Vertical and Horizontal polarization
            [1.4, 6.9, 10.65, 18.7, 37],                 # Frequency labels in GHz
            'CIMR'                                      # Sensor name
        )
        return CIMR

    elif name == "MWI":
        # Define MWI sensor with its frequencies, incidence angle, and polarizations
        MWI = sensor_list.passive(
            [18.7e9, 23.8e9, 31.4e9, 50.3e9, 52.7e9, 53.24e9, 53.75e9, 89e9], 
            53.1,                                       # Incidence angle in degrees
            ['V', 'H'],                                 # Vertical and Horizontal polarization
            [18.7, 23.8, 31.4, 50.3, 52.7, 53.24, 53.75, 89],  # Frequency labels in GHz
            'MWI'                                       # Sensor name
        )
        return MWI

    else:
        print('Invalid sensor name')
        return None

