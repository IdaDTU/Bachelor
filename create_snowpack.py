from smrt import make_snowpack
import numpy as np
from thermal_gradients import compute_thermal_gradients

def create_snowpack(number_of_layers,
                    thickness_ice,
                    thickness_snow,
                    temperature_air,
                    temperature_water):
    """
    Function to create a snowpack using SMART's make_snowpack function.
    
    Args:
        thickness (float): The thickness of the snowpack in meters.
        corr_length (float): The correlation length of the snowpack in meters.
        microstructure_model (str): The model for the snow microstructure.
        density (float): The density of the snow in kg/m³.
        substrate (str): The type of substrate the snow is on.
        temperature (float): The temperature of the snowpack in °C.
        ice_permittivity_model (str): The model for the ice permittivity.
        background_permittivity_model (str): The model for the background permittivity.
        volumetric_liquid_water (float): The volumetric liquid water content (in %).
        liquid_water (float): The liquid water content in the snowpack.
        salinity (float): The salinity of the snowpack (in ppt).
        medium (str): The surrounding medium (e.g., air, etc.).

    Returns:
        object: The snowpack object created with the provided specifications.
    """
    
    n = number_of_layers
    hi = np.array([thickness_ice/n] * n).T 
    hs = np.array([thickness_snow/n] * n).T
    
    #density = np.full_like(thickness_snow, 300, dtype=float)
    
    # Store all temperature profiles
    #temperature_profiles = []

    # Compute temperature profile for each point
    #for i in range(len(thickness_snow)):
     #   depths = np.linspace(0, hs[i], n, dtype=np.float64)  # Define depth array
      #  temperatures = compute_thermal_gradients(
       #     temperature_air.iloc[i], 
        #    temperature_water.iloc[i], 
         #   hi[i], 
          #  hs[i],
           # depths
        #)
        #temperature_profiles.append(temperatures)  # Store the list

    # Convert temperature_profiles into a numpy array (600, 5)
   # temperature_profiles = np.array(temperature_profiles)
    
    thickness_snow = np.array([0.05, 0.2])
    temperature_s = np.linspace(273.15-25., 273.15 - 20, n)
    density_s = [200, 340]
    
    
    return make_snowpack(thickness=thickness_snow,
                         corr_length=np.array([5e-5]*n),
                         microstructure_model='exponential',
                         density=density_s,
                         temperature=temperature_s,
                         salinity=0,
                         medium='snow',
                         inclusion_shape = 'spheres',
                         emmodel_options={'inclusion_shape': 'spheres'})

