import numpy as np
import pandas as pd
from smrt import make_ice_column, make_snowpack, make_model, sensor_list

def SMRT_create_ice_columns(thickness_ice,
                temperature_profile_ice,
                salinity_profile_ice,
                n,
                ice_type,
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
                          denisty_profile_snow,
                          n,
                          layers_snow):
    snowpacks = []
    for i in range(n):    
        l_s=layers_snow #3 snow layers
        print(f"Processing snowpacks index: {i}. Finished: {(i/n)}...")
        thickness_s = np.linspace(0, thickness_snow.iloc[i], l_s)
        p_ex_s = np.linspace(0.000054, 0.000442, l_s) 
        temperature_s = temperature_profile_snow.iloc[i]
        
        density_s = denisty_profile_snow.iloc[i]
        
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

def SMRT_compute_tbv(mediums, freq, theta):
    """
    Computes the brightness temperature in vertical polarization for a given list of mediums.
    
    Parameters:
    - mediums: list of medium objects to process
    - freq: Frequency in Hz (e.g., 37e9 for 37 GHz)
    - theta: Incidence angle in degrees
    
    Returns:
    - DataFrame with computed brightness temperatures (tbv).
    """
    results = []

    for i, medium in enumerate(mediums):
        print(f"Processing index: {i}")
        
        # Create the sensor object for the given frequency and incidence angle.
        sensor = sensor_list.passive(freq, theta)
        
        # Increase the number of streams for a more accurate TB calculation
        n_max_stream = 128  # Default is 32; increase if using > 1 snow layer on top of sea ice.
        m = make_model("iba", "dort", rtsolver_options={"n_max_stream": n_max_stream})
        
        # Run the model for snow-covered sea ice:
        res = m.run(sensor, medium)
        
        # Compute the brightness temperature in vertical polarization.
        tbh = res.TbH()
        results.append(tbh)

    # Create a DataFrame to store the results.
    tbH_df = pd.DataFrame({'tbh': results})
    return tbH_df

