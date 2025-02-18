import numpy as np
import pandas as pd
from smrt import make_ice_column, make_snowpack, make_model, sensor_list

def SMRT_create_ice_columns(thickness_ice,
                temperature_profile_ice,
                salinity_profile_ice,
                n):
    
    ice_columns = []
    # Loop over each index to create the ice columns.
    for i in range(n):
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
                          temperature_snow,
                          n):
    snowpacks = []
    for i in range(n):    
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
    return snowpacks
    
def SMRT_create_mediums(snowpacks,
                       ice_columns,
                       n):
    mediums=[]
    for i in range(n):
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

def SMRT_calculate_brightness_temperature(sensor,
                                          mediums):
       # Define CIMR sensor with specified frequencies, incidence angle, and polarizations
        CIMR = sensor_list.passive([1.4e9, 6.9e9, 10.65e9, 18.7e9, 36.5e9],  # Frequencies in Hz
                                   55,  # Incidence angle in degrees
                                   ['V', 'H'],  # Vertical and Horizontal polarization
                                   [1.4, 6.9, 10.65, 18.7, 37],  # Frequency labels in GHz
                                   'CIMR')  # Sensor name

        # Define MWI sensor with its frequencies, incidence angle, and polarizations
        MWI = sensor_list.passive([18.7e9, 23.8e9, 31.4e9, 50.3e9, 52.7e9, 53.24e9, 53.75e9, 89e9], 
                                   53.1,  # Incidence angle in degrees
                                   ['V', 'H'],  # Vertical and Horizontal polarization
                                   [18.7, 23.8, 31.4, 50.3, 52.7, 53.24, 53.75, 89],  # Frequency labels in GHz
                                   'MWI')  # Sensor name
        
        # Run the model and retrieve brightness temperature values based on the chosen sensor
        if sensor == 'CIMR':
            m = make_model("iba", "dort")  # Create model using "iba" and "dort" solvers
            result = m.run(CIMR, mediums)  # Run the model simulation
            
            # Store the results in a DataFrame
            df = {
                'Channel': ['1.4GHz', '6.9GHz', '10.65GHz', '18.7GHz', '37GHz'],
                'TB(V)': result.TbV(),  # Brightness Temperature for Vertical polarization
                'TB(H)': result.TbH(),  # Brightness Temperature for Horizontal polarization
                'PR': result.polarization_ratio()  # Polarization Ratio
            }
            df = pd.DataFrame(df)
        
        elif sensor == 'MWI':
            m = make_model("iba", "dort")  # Create model using "iba" and "dort" solvers
            result = m.run(MWI, mediums)  # Run the model simulation
            
            # Store the results in a DataFrame
            df = {
                'Channel': ['18.7GHz', '23.8GHz', '31.4GHz', '50.3GHz', '52.7GHz', '53.24GHz', '53.75GHz', '89GHz'],
                'TB(V)': result.TbV(),  # Brightness Temperature for Vertical polarization
                'TB(H)': result.TbH(),  # Brightness Temperature for Horizontal polarization
                'PR': result.polarization_ratio()  # Polarization Ratio
            }
            df = pd.DataFrame(df)
        
        else:
            # Raise an error if an invalid sensor is specified
            raise ValueError('Invalid sensor. Use "CIMR" or "MWI".')
        
        return df  # Return the DataFrame containing the results
