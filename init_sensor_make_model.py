from smrt import make_snowpack, make_ice_column, make_model, sensor_list
import pandas as pd

def init_sensor_snowpack(sensor,
                         thickness,
                         corr_length,
                         microstructure_model,
                         density,
                         temperature=273.15,
                         ice_permittivity_model=None,
                         background_permittivity_model=1.0,
                         volumetric_liquid_water=None,
                         liquid_water=None,
                         salinity=0,
                         medium='snow'):
    """
    Initializes a snowpack model and simulates brightness temperatures for a given sensor.

    Parameters:
    - sensor (str): The sensor to use ('CIMR' or 'MWI').
    - thickness (float): Thickness of the snow layer (m).
    - corr_length (float): Correlation length of snow microstructure (m).
    - microstructure_model (str): Model used to define the snow microstructure.
    - density (float): Snow density (kg/m³).
    - temperature (float, optional): Temperature of the snowpack (K). Default is 273.15 K.
    - ice_permittivity_model (str, optional): Model for ice permittivity. Default is None.
    - background_permittivity_model (float, optional): Background permittivity. Default is 1.0.
    - volumetric_liquid_water (float, optional): Volumetric liquid water fraction. Default is None.
    - liquid_water (float, optional): Liquid water content in snow. Default is None.
    - salinity (float, optional): Salinity content in snow (ppt). Default is 0.
    - medium (str, optional): Medium type (default is 'snow').

    Returns:
    - pd.DataFrame: A DataFrame containing brightness temperatures (TB) for vertical and horizontal polarization,
                    along with the polarization ratio (PR) for the chosen sensor.
    """
    
    # Create a snowpack model with the given properties
    snowpack = make_snowpack(thickness=thickness,
                             corr_length=corr_length,
                             microstructure_model=microstructure_model,
                             density=density,
                             temperature=temperature,
                             ice_permittivity_model=ice_permittivity_model,
                             background_permittivity_model=background_permittivity_model, 
                             volumetric_liquid_water=volumetric_liquid_water, 
                             liquid_water=liquid_water,
                             salinity=salinity,
                             medium=medium)
    
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
        result = m.run(CIMR, snowpack)  # Run the model simulation
        
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
        result = m.run(MWI, snowpack)  # Run the model simulation
        
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

######################################################################################################################

def init_sensor_icecolumn(sensor,
                          ice_type,
                          thickness,
                          temperature,
                          microstructure_model,
                          corr_length,
                          brine_inclusion_shape='spheres',
                          salinity=0.0, 
                          brine_volume_fraction=None, 
                          brine_volume_model=None,
                          brine_permittivity_model=None,
                          ice_permittivity_model=None, 
                          saline_ice_permittivity_model=None,
                          porosity=0, 
                          density=None, 
                          add_water_substrate=True,
                          surface=None, 
                          interface=None,
                          substrate=None,
                          atmosphere=None):
    
    """ 
    Initializes an ice column and simulates microwave brightness temperatures (TB) 
    using the SMRT model for either CIMR or MWI satellite sensors.
    
    Parameters:
    - sensor: 'CIMR' or 'MWI' (str)
    - ice_type: Type of sea ice (str)
    - thickness: Ice thickness in meters (float)
    - temperature: Ice temperature profile (list or float)
    - microstructure_model: Ice microstructure model (str)
    - brine_inclusion_shape: Shape of brine inclusions (default='spheres')
    - salinity: Ice salinity (default=0.0)
    - porosity: Ice porosity (default=0)
    - density: Ice density (default=None)
    - add_water_substrate: Include water substrate (default=True)
    - Other optional parameters for defining ice and environmental properties.

    Returns:
    - Pandas DataFrame with TB(V), TB(H), and polarization ratio (PR) for each channel.
    """

    # Create ice column model
    icecolumn = make_ice_column(ice_type=ice_type,
                                thickness=thickness,
                                temperature=temperature,
                                microstructure_model=microstructure_model,
                                corr_length = corr_length,
                                brine_inclusion_shape=brine_inclusion_shape,
                                salinity=salinity,
                                brine_volume_fraction=brine_volume_fraction,
                                brine_volume_model=brine_volume_model,
                                brine_permittivity_model=brine_permittivity_model,
                                ice_permittivity_model=ice_permittivity_model,
                                saline_ice_permittivity_model=saline_ice_permittivity_model,
                                porosity=porosity,
                                density=density,
                                add_water_substrate=add_water_substrate,
                                surface=surface,
                                interface=interface,
                                substrate=substrate,
                                atmosphere=atmosphere
    )

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
         result = m.run(CIMR, icecolumn)  # Run the model simulation
         
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
         result = m.run(MWI, icecolumn) # Run the model simulation
         
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