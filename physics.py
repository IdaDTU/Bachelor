import numpy as np
import pandas as pd

def temperature_gradient_snow(thickness_snow, 
                              thickness_ice, 
                              temperature_air,
                              layers_snow):
    """
    Calculate the temperature profile within the snow layer for each set of inputs.

    Parameters:
        thickness_snow (pd.Series): Series of snow layer thicknesses in meters.
        thickness_ice (pd.Series): Series of ice layer thicknesses in meters.
        temperature_air (pd.Series): Series of air temperatures in Kelvin.
        layers_snow (int): Number of layers to divide the snow thickness into.

    Returns:
        pd.Series: Each element is a NumPy array of temperatures.
    """
    # Constants
    ksi = 2.1  # Ice thermal conductivity (W/m/K)
    ks = 0.3   # Snow thermal conductivity (W/m/K)
    temperature_water = 271.15  # Water temperature (K)
    
    # Compute the temperature at the ice-snow interface for each row
    interface_temp = (ksi * thickness_snow * temperature_water + ks * thickness_ice * temperature_air) / \
                     (ksi * thickness_snow + ks * thickness_ice)
    
    # Calculate the thermal gradient in the snow for each row
    gradient = (interface_temp - temperature_air) / thickness_snow

    # Generate snow layer temperatures for each row
    profiles = []
    for i in range(len(thickness_snow)):
        depths = np.linspace(0, thickness_snow.iloc[i], layers_snow)  # Evenly spaced depth levels
        profile = temperature_air.iloc[i] + gradient.iloc[i] * depths  # Temperature at each layer depth
        profiles.append(profile)  # Append the NumPy array directly
    
    # Create a Series with snow temperature profiles
    return pd.Series(profiles)

def calculate_snow_density(thickness_snow, layers_snow):
    """
    Calculate snow density as a function of depth, where depths are derived from layers.

    Parameters:
    - thickness_snow (pd.Series): Snow thickness values (m).
    - layers_snow (int): Number of depth layers to compute.

    Returns:
    - np.array: Snow density at each depth for each snow thickness.
    """
    density_bulk = 320  # Bulk density of snow (kg/m³)
    density_water = 1000  # Density of water (kg/m³)

    densities_snow = []

    for i in range(len(thickness_snow)):
        SWE = thickness_snow.iloc[i] * density_bulk  # Snow Water Equivalent (kg/m²)
        print(SWE)
        # Generate depth levels based on the number of layers
        depths = np.linspace(0, thickness_snow.iloc[i], layers_snow)  
        print(depths)

        density_profile = []
        
        for depth in depths:
            if depth == 0:
                density_snow = density_bulk  # Surface density (initial value)
            else:
                density_snow = (density_water * SWE) / (depth * 10)  # Avoid division by zero
            
            density_profile.append(density_snow)
        
        densities_snow.append(density_profile)

    return pd.Series(densities_snow) 










