from smrt import make_snowpack, make_ice_column, make_model, sensor_list

def create_sensor(sensor_list, frequencies, angle, polarizations, labels, name):
    """
    Function to create a sensor with specified frequencies, incidence angle,
    polarizations, labels, and name.
    
    Args:
        sensor_list (object): The list or framework to hold the sensor.
        frequencies (list): List of frequencies in Hz.
        angle (float): The incidence angle in degrees.
        polarizations (list): List of polarizations (e.g., 'V' for vertical, 'H' for horizontal).
        labels (list): List of frequency labels in GHz.
        name (str): The name of the sensor.
        
    Returns:
        object: A sensor object created with the provided specifications.
    """
    return sensor_list.passive(frequencies, angle, polarizations, labels, name)
