from smrt import make_snowpack, make_ice_column, make_model, sensor_list

def create_snowpack(thickness,
                    density,
                    substrate,                    
                    temperature,
                    medium):
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
    return make_snowpack(thickness=thickness,
                         corr_length=np.repeat(5.0e-5 * thickness_snow.values.reshape(-1, 1), n, axis=1),
                         microstructure_model='exponential',
                         density=density,
                         substrate=substrate,
                         temperature=temperature,
                         salinity=0,
                         medium=medium)


