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


def firn_densification(accumulation_rate, thickness_snow, rho_0, snow_layers):
    """
    Calculate the firn density profile as a function of snow layers.

    Parameters:
    - accumulation_rate: Array of annual snow accumulation rates (m water equivalent)
    - thickness_snow: Array of snow thickness values (m)
    - rho_0: Initial snow density at the surface (Mg/m^3)
    - snow_layers: Number of snow layers (each layer represents 1 year)

    Returns:
    - depths: Array of depths (m)
    - densities: Array of densities (Mg/m^3)
    """
    
    # Constants
    rho_i = 0.917  # Density of ice in Mg/m^3
    
    # Stage 1 constants
    a = 1.1  # from the article, empirical constant for Stage 1
    b = 0.5  # from the article, empirical constant for Stage 2

    # Initialize arrays for depths and densities
    depths = thickness_snow
    densities = np.zeros_like(depths)

    # Stage 1: Densification for rho < 0.55 Mg/m^3
    for i in range(snow_layers):
        # Densification based on the current accumulation rate and snow thickness
        if rho_0 + a * accumulation_rate[i] * depths[i] < 0.55:
            densities[i] = rho_0 + a * accumulation_rate[i] * depths[i]
        else:
            # Transition to Stage 2 densification
            stage1_end_depth = (0.55 - rho_0) / (a * accumulation_rate[i])
            if depths[i] < stage1_end_depth:
                densities[i] = rho_0 + a * accumulation_rate[i] * depths[i]
            else:
                densities[i] = 0.55 + b * accumulation_rate[i]**0.5 * (depths[i] - stage1_end_depth)

    return densities*1000










