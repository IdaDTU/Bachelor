from smrt import make_model, sensor_list
import pandas as pd
import numpy as np


def compute_smrt_tb(sensor,frequency,medium):
  
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



    for layer in medium.layers:
        if not hasattr(layer, 'inclusion_shape') or layer.inclusion_shape not in ['spheres', 'random_needles']:
            print(f"⚠️ Warning: Layer {layer} has invalid inclusion_shape. Setting to 'spheres'.")
            layer.inclusion_shape = 'spheres'  # Default to spheres if missing or incorrect


    # Run the model and retrieve brightness temperature values based on the chosen sensor
    if sensor == 'CIMR':
        m = make_model("iba", "dort", rtsolver_options=dict(error_handling='nan', 
                                                            diagonalization_method='shur_forcedtriu', 
                                                            phase_normalization=True, 
                                                            n_max_stream=128))  # Create model using "iba" and "dort" solvers
        result = m.run(CIMR, medium)  # Run the model simulation
        
        # Store the results in a DataFrame
        df = {
            'Channel': ['1.4GHz', '6.9GHz', '10.65GHz', '18.7GHz', '37GHz'],
            'TB(V)': result.TbV(),  # Brightness Temperature for Vertical polarization
            'TB(H)': result.TbH(),  # Brightness Temperature for Horizontal polarization
            'PR': result.polarization_ratio()  # Polarization Ratio
        }
        
        df = pd.DataFrame(df)
        
        # Mapping user input frequency to actual frequency labels
        freq_map = {
            '1.4': '1.4GHz',
            '6.9': '6.9GHz',
            '10': '10.65GHz',
            '19': '18.7GHz',
            '37': '37GHz'
            }

        # Debugging check: is frequency in freq_map?
        if frequency not in freq_map:
            print(f"⚠️ Invalid frequency: {frequency}. Choose one of {list(freq_map.keys())}")
            return None

        # Debugging check: does the DataFrame contain this channel?
        if freq_map[frequency] not in df['Channel'].values:
            print(f"⚠️ Frequency {freq_map[frequency]} not found in DataFrame!")
            return None

        # Filter by selected frequency
        df = df[df['Channel'] == freq_map[frequency]]
    
    elif sensor == 'MWI':
        m = make_model("iba", "dort", rtsolver_options=dict(error_handling='nan', 
                                                            diagonalization_method='shur_forcedtriu', 
                                                            phase_normalization=True, 
                                                            n_max_stream=128))  # Create model using "iba" and "dort" solvers
        result = m.run(MWI, medium)  # Run the model simulation
        
        # Store the results in a DataFrame
        df = {
            'Channel': ['18.7GHz', '23.8GHz', '31.4GHz', '50.3GHz', '52.7GHz', '53.24GHz', '53.75GHz', '89GHz'],
            'TB(V)': result.TbV(),  # Brightness Temperature for Vertical polarization
            'TB(H)': result.TbH(),  # Brightness Temperature for Horizontal polarization
            'PR': result.polarization_ratio()  # Polarization Ratio
        }
        df = pd.DataFrame(df)
        
        # Mapping user input frequency to actual frequency labels
        freq_map = {
            '19': '18.7GHz',
            '24': '23.8GHz',
            '31': '31.4GHz',
            '50': '50.3GHz',
            '53': '52.7GHz',
            '53.24': '53.24GHz',
            '53.75': '53.75GHz',
            '89': '89GHz',
            }

        # Debugging check: is frequency in freq_map?
        if frequency not in freq_map:
            print(f"⚠️ Invalid frequency: {frequency}. Choose one of {list(freq_map.keys())}")
            return None

        # Debugging check: does the DataFrame contain this channel?
        if freq_map[frequency] not in df['Channel'].values:
            print(f"⚠️ Frequency {freq_map[frequency]} not found in DataFrame!")
            return None

        # Filter by selected frequency
        df = df[df['Channel'] == freq_map[frequency]]
        
    
    else:
        # Raise an error if an invalid sensor is specified
        raise ValueError('Invalid sensor. Use "CIMR" or "MWI".')
    
    
    return df







     
     
     
     
     
     