from smrt import make_snowpack, make_ice_column
from thermal_gradients import compute_thermal_gradients
import numpy as np


def create_icecolumn(ice_type,
                      number_of_layers,
                      thickness_ice,
                      thickness_snow,
                      temperature_air,
                      temperature_water,
                      temperature_ice):
    """
    Function to create an ice column using SMART's make_ice_column function.
    
    Args:
        ice_type (str): The type of ice (e.g., 'fresh', 'sea', etc.).
        thickness (float): The thickness of the ice column in meters.
        temperature (float): The temperature of the ice column in °C.
        microstructure_model (str): The model for the ice microstructure.
        corr_length (float): The correlation length of the ice column in meters.
        brine_inclusion_shape (str): The shape of the brine inclusions in the ice.
        salinity (float): The salinity of the ice (in ppt).
        brine_volume_fraction (float): The fraction of brine volume in the ice.
        brine_volume_model (str): The model for brine volume.
        brine_permittivity_model (str): The model for brine permittivity.
        ice_permittivity_model (str): The model for ice permittivity.
        saline_ice_permittivity_model (str): The model for saline ice permittivity.
        porosity (float): The porosity of the ice.
        density (float): The density of the ice in kg/m³.
        add_water_substrate (bool): Whether to add a water substrate under the ice.
        surface (str): The type of surface on the top of the ice column.
        interface (str): The type of interface at the bottom of the ice column.
        substrate (str): The type of substrate beneath the ice column.
        atmosphere (str): The atmosphere surrounding the ice column.
        
    Returns:
        object: The ice column object created with the provided specifications.
    """
    
    n = number_of_layers
    hi = np.array([thickness_ice/n] * n).T 
    hs = np.array([thickness_snow/n] * n).T
    
    # Store all temperature profiles
    temperature_profiles = []

    # Compute temperature profile for each point
    for i in range(len(thickness_ice)):
        depths = np.linspace(0, hi.iloc[i], n, dtype=np.float64)  # Define depth array
        temperatures = compute_thermal_gradients(
            temperature_air.iloc[i], 
            temperature_water.iloc[i], 
            hi.iloc[i], 
            hs.iloc[i],
            depths
        )
        temperature_profiles.append(temperatures)  # Store the list

    # Convert temperature_profiles into a numpy array (600, 5)
    temperature_profiles = np.array(temperature_profiles)
    
    
    salinity = np.repeat(7.88 - 1.59 * hi.values.reshape(-1, 1), n, axis=1)
    porosity = salinity * (0.05322 - 4.919 / temperature_ice.values.reshape(-1, 1)) 
    
    return make_ice_column(ice_type=ice_type,
                           thickness=thickness_ice,
                           temperature=temperature_profiles,
                           microstructure_model='exponential',
                           corr_length=np.repeat(1.0e-3 * thickness_ice.values.reshape(-1, 1), n, axis=1),
                           brine_inclusion_shape='sphere',
                           salinity=salinity,
                           brine_volume_fraction=0.0,
                           porosity=porosity,
                           add_water_substrate=True)
