import os
import glob
import xarray as xr
import numpy as np
import pandas as pd

def combine_nc_files(directory):
    """
    Combine all .nc files in the specified directory into one xarray.Dataset.
    
    Parameters:
        directory (str): The directory containing the .nc files.
    
    Returns:
        xarray.Dataset: The combined dataset.
    """
    # Create a file pattern that matches all .nc files in the directory
    file_pattern = os.path.join(directory, '*.nc')
    
    # Get a sorted list of .nc files
    nc_files = sorted(glob.glob(file_pattern))
    
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in directory: {directory}")
    
    # Open and combine the datasets along matching coordinates
    combined_ds = xr.open_mfdataset(nc_files, combine='by_coords')
    return combined_ds

def create_input_dataframe(ds,
                           layers_ice):
    """
    Create a Pandas DataFrame from all variables in the dataset
    without applying any manual indexing. All dimensions (e.g., time,
    spatial coordinates) will become columns in the DataFrame.
    
    Rows where TLAT is below the Arctic Circle (i.e., TLAT < 66.5°) are dropped.

    Parameters:
        ds (xarray.Dataset): The dataset containing multiple variables.
    
    Returns:
        pd.DataFrame: A DataFrame with all variables and coordinate labels,
                      with rows corresponding to every combination of coordinates.
    """
    
    # Select only the desired variables from the dataset.
    ds = ds[['TLAT', 'TLON', 'hi', 'hs', 'Sinz', 'Tinz','Tsnz','Tair']]
    
    # Convert the dataset to a DataFrame and reset the index so that all coordinates become columns.
    df = ds.to_dataframe().reset_index()
    
    # Convert Tinz, Tair and Tsnz from °C to K and Sinz from ppt to kg/kg.
    df['tinz'] = df['Tinz'] + 273.15
    df['tsnz'] = df['Tsnz'] + 273.15
    df['tair'] = df['Tair'] + 273.15
    df['sinz'] = df['Sinz'] / 1000
    
    # Drop orginal columns
    df = df.drop(columns=['Tinz', 'Tsnz', 'Tair', 'Sinz'])

    
    # Filter rows: Keep only rows where TLAT is within the Arctic Circle.
    df = df[df['TLAT'] >= 67]
    
    # Drop NaN
    df = df.dropna()
        
    # Group by the measurement-defining columns, including 'nc'
    group_cols = ['TLAT', 'TLON', 'hi', 'hs','tsnz','tair','nc']
    
    # First aggregate as lists, only containing 3 layers
    profiles_df = (
        df.groupby(group_cols)
          .agg({
              'tinz': lambda x: list(x),
              'sinz': lambda x: list(x)
          })
          .reset_index()
          .rename(columns={
              'tinz': 'temperature_profiles',
              'sinz': 'salinity_profiles'
          })
    )
    
    # Now convert the list values into NumPy arrays
    profiles_df['temperature_profiles'] = profiles_df['temperature_profiles'].apply(np.array)
    profiles_df['salinity_profiles'] = profiles_df['salinity_profiles'].apply(np.array)


    # Acces n layers from top (first three values in each row)
    profiles_df['temperature_profiles'] = profiles_df['temperature_profiles'].apply(lambda x: x[:layers_ice])
    profiles_df['salinity_profiles'] = profiles_df['salinity_profiles'].apply(lambda x: x[:layers_ice])
            
    return profiles_df


#
def remove_nonphysical(df,
                       layers_ice):
    # Drop rows where 'tsnz' exceeds 273.15K.
    df = df[df['tsnz'] <= 273.15]
    
    # Initialize a set to store indices of rows to drop
    dropped_rows = set()

    # Check temperature_profiles: drop row if any of the first 7 elements is > 273.15
    for i in range(len(df['temperature_profiles'])):
        for j in range(layers_ice):
            if df['temperature_profiles'].iloc[i][j] > 273.15:
                dropped_rows.add(df.index[i])
                break  # Once a violation is found, exit inner loop
    
    # Check salinity_profiles: drop row if any of the first 7 elements is < 2
    for i in range(len(df['salinity_profiles'])):
        for j in range(layers_ice):
            if df['salinity_profiles'].iloc[i][j] < 0.002: 
                dropped_rows.add(df.index[i])
                break  # Exit inner loop on violation
    
    # Drop the identified rows from the DataFrame
    df = df.drop(list(dropped_rows))

    return df



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
        layers (int): Number of layers to divide the snow thickness into.

    Returns:
        pd.Series: Series where each element is a NumPy array of temperatures.
    """
    # Constants
    ksi = 2.1  # Ice thermal conductivity (W/m/K)
    ks = 0.3   # Snow thermal conductivity (W/m/K)
    temperature_water = 271.15  # Water temperature (K)
    
    # Compute the temperature at the ice-snow interface for each row
    interface_temp = (ksi * thickness_snow * temperature_water + ks * thickness_ice * temperature_air) / (ksi * thickness_snow + ks * thickness_ice)
    
    # Calculate the thermal gradient in the snow for each row
    gradient = (interface_temp - temperature_air) / thickness_snow

    # Generate snow layer temperatures for each row
    profiles = []
    for i in range(len(thickness_snow)):
        depths = np.linspace(0, thickness_snow.iloc[i], layers_snow)  # Evenly spaced depth levels
        profile = temperature_air.iloc[i] + gradient.iloc[i] * depths  # Temperature at each layer depth
        profiles.append(profile)  # Append the NumPy array directly
    
    # Create a Series with snow temperature profiles
    profile_series = pd.Series(profiles)

    return profile_series
