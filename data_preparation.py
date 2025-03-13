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


def create_input_dataframe(ds, layers_ice):
    """
    Create a DataFrame with temperature and salinity profiles.
    Selects the rows with the maximum 'aicen' per group, but collects tinz/sinz profiles from all rows.
    
    Parameters:
        ds (xarray.Dataset): Dataset with sea ice variables.
        layers_ice (int): Number of layers to include in the profiles.
    
    Returns:
        pd.DataFrame: One row per group with profiles + features.
    """
    
    # Select relevant variables
    ds = ds[['TLAT', 'TLON', 'hi', 'hs', 'Sinz', 'Tinz', 'Tsnz', 'Tair', 'snow', 'nc']]
    
    # Convert to DataFrame
    df = ds.to_dataframe().reset_index()

    # Convert units
    df['tinz'] = df['Tinz'] + 273.15
    df['tsnz'] = df['Tsnz'] + 273.15
    df['tair'] = df['Tair'] + 273.15
    df['sinz'] = df['Sinz'] / 1000

    # Drop originals
    df.drop(columns=['Tinz', 'Tsnz', 'Tair', 'Sinz'], inplace=True)

    # Clean data
    df = df.dropna()
    df = df[df['TLAT'] >= 67]

    # Grouping columns (excluding nc + aicen because they vary across layers)
    group_cols = ['TLAT', 'TLON', 'hi', 'hs', 'tsnz', 'tair', 'snow']

    ### Step 1: Select the row with the max 'aicen' (as the representative row)
    idx_max_aicen = df.groupby(group_cols)['aicen'].idxmax()
    df_max_aicen = df.loc[idx_max_aicen].reset_index(drop=True)

    ### Step 2: Collect tinz and sinz into profiles (lists of values per group)
    profiles = (
        df.groupby(group_cols)
          .agg({
              'tinz': lambda x: list(x)[:layers_ice],  # Collect up to n layers
              'sinz': lambda x: list(x)[:layers_ice]
          })
          .reset_index()
    )

    # Step 3: Merge the max aicen info with the profiles
    merged_df = pd.merge(df_max_aicen, profiles, on=group_cols, suffixes=('', '_profile'))

    # Step 4: Rename for clarity (optional)
    merged_df = merged_df.rename(columns={
        'tinz_profile': 'temperature_profiles',
        'sinz_profile': 'salinity_profiles'
    })

    # Convert lists to numpy arrays
    merged_df['temperature_profiles'] = merged_df['temperature_profiles'].apply(np.array)
    merged_df['salinity_profiles'] = merged_df['salinity_profiles'].apply(np.array)

    return merged_df



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
