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
    ksi = 2  # Ice thermal conductivity (W/m/K)
    ks = 0.25   # Snow thermal conductivity (W/m/K)
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
        profile = np.minimum(profile, 273.15)  # Cap temperatures at 273.15 K
        profiles.append(profile)
    
    return pd.Series(profiles)


# def calculate_snow_density(temperature_snow, thickness_snow, layers_snow):
#     """
#     Calculate the snow density profile within the snow layer for each set of inputs
#     using a thermal metamorphism model.

#     Parameters:
#         temperature_snow (pd.Series): Series of snow temperatures in Kelvin.
#         thickness_snow (pd.Series): Series of snow layer thicknesses in meters.
#         layers_snow (int): Number of layers to divide the snow thickness into.

#     Returns:
#         pd.Series: Each element is a NumPy array of densities (kg/m³).
#     """
    
#     # Constants
#     density_surface = 300 / 1000   # Mg/m³
#     density_ice = 917 / 1000       # Mg/m³
#     R = 8.314                     # J/mol·K
    
#     profiles = []
    
#     # Iterate over each row of input data
#     for i in range(len(thickness_snow)):
        
#         depths = np.linspace(0, thickness_snow.iloc[i], layers_snow)  # Evenly spaced depth levels
        
#         # Metamorphism rate constant (k0)
#         k0 = 11 * np.exp(-10160 / (R * temperature_snow.iloc[i]))
        
#         # Calculate Z0 at each depth
#         Z0 = np.exp(density_ice * k0 * depths + np.log(density_surface / (density_ice - density_surface)))
        
#         # Calculate density profile (convert back to kg/m³)
#         density_profile = (density_ice * Z0) / (1 + Z0) * 1000
        
#         profiles.append(density_profile)  # Append NumPy array directly
    
#     # Return as a pandas Series, matching the other function's style
#     return pd.Series(profiles)



def calculate_snow_salinity(snow_thickness, snow_layers, ice_type):
    """
    Calculate salinity of snow depending on depth and ice type.

    Parameters:
    snow_thickness : pd.Series
        Series of snow layer thicknesses in meters.
    snow_layers : int
        Number of layers to divide the snow thickness into.
    ice_type : str
        'firstyear' for first-year ice or 'multiyear' for multi-year ice.

    Returns:
    pd.Series
        Each element is a NumPy array of salinity values in psu.
    """
    salinity_profiles = []
    
    for thickness in snow_thickness:
        depths = np.linspace(0, thickness * 100, snow_layers)  # Convert m to cm

        if ice_type == 'firstyear':
            salinity = np.zeros_like(depths)
            for i, depth in enumerate(depths):
                if depth <= 10:
                    salinity[i] = 10/1000  # psu to kg/kg
                elif 10 < depth <= 30:
                    salinity[i] = 1/1000   #  psu to kg/kg
                else:
                    salinity[i] = 0.1/1000 #  psu to kg/kg

        elif ice_type == 'multiyear':
            salinity = np.full_like(depths, 0.01/1000)  #  psu to kg/kg

        else:
            raise ValueError("ice_type must be 'firstyear' or 'multiyear'")
        
        salinity_profiles.append(salinity)
    
    return pd.Series(salinity_profiles)

