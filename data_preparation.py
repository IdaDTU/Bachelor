import os
import glob
import xarray as xr
import numpy as np

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

def create_combined_dataframe(ds, layers_ice):
    """
    Create a DataFrame containing sea ice temperature and salinity profiles.
    It selects one representative row per unique location+ice state group based on aggregated profiles.

    Parameters:
        ds (xarray.Dataset): Dataset with sea ice variables.
        layers_ice (int): Number of ice layers to include in the temperature and salinity profiles.

    Returns:
        pd.DataFrame: A DataFrame with one row per group including metadata and the temperature/salinity profiles.
    """
    # Keep only the relevant variables from the dataset
    ds = ds[['TLAT', 'TLON', 'hi', 'hs', 'Sinz', 'Tinz', 'Tsnz', 'Tair', 'snow', 'nc','aice','iage']]

    # Convert to pandas DataFrame
    df = ds.to_dataframe().reset_index()

    # Convert units
    df['tinz'] = df['Tinz'] + 273.15
    df['tsnz'] = df['Tsnz'] + 273.15
    df['tair'] = df['Tair'] + 273.15
    df['sinz'] = df['Sinz'] / 1000

    # Drop originals
    df.drop(columns=['Tinz', 'Tsnz', 'Tair', 'Sinz'], inplace=True)

    # Clean and filter
    df = df.dropna()
    df = df[df['TLAT'] >= 55.5]

    # Define grouping
    group_cols = ['TLAT', 'TLON', 'hi', 'hs', 'tsnz', 'tair', 'snow','aice','iage']

    # Aggregate into profiles
    grouped = df.groupby(group_cols).agg({'tinz': lambda x: list(x)[:layers_ice],
                                          'sinz': lambda x: list(x)[:layers_ice] }).reset_index()

    # Rename and convert to numpy arrays
    grouped = grouped.rename(columns={'tinz': 'temperature_profiles',
                                      'sinz': 'salinity_profiles' })

    grouped['temperature_profiles'] = grouped['temperature_profiles'].apply(np.array)
    grouped['salinity_profiles'] = grouped['salinity_profiles'].apply(np.array)

    return grouped


def create_combined_dataframe(ds, layers_ice):
    """
    Create a DataFrame containing sea ice temperature and salinity profiles.
    It selects one representative row per unique location+ice state group based on aggregated profiles.
    Also calculates the average temperature and salinity per profile.

    Parameters:
        ds (xarray.Dataset): Dataset with sea ice variables.
        layers_ice (int): Number of ice layers to include in the temperature and salinity profiles.

    Returns:
        pd.DataFrame: A DataFrame with one row per group including metadata,
                      the temperature/salinity profiles, and their averages.
    """
    # Keep only the relevant variables
    ds = ds[['TLAT', 'TLON', 'hi', 'hs', 'Sinz', 'Tinz', 'Tsnz', 'Tair', 'snow', 'nc', 'aice', 'iage']]

    # Convert to pandas DataFrame
    df = ds.to_dataframe().reset_index()

    # Convert units
    df['tinz'] = df['Tinz'] + 273.15  # Celsius to Kelvin
    df['tsnz'] = df['Tsnz'] + 273.15
    df['tair'] = df['Tair'] + 273.15
    df['sinz'] = df['Sinz'] / 1000    # PSU to fraction

    # Drop originals
    df.drop(columns=['Tinz', 'Tsnz', 'Tair', 'Sinz'], inplace=True)

    # Clean and filter
    df = df.dropna()
    df = df[df['TLAT'] >= 55.5]

    # Define grouping
    group_cols = ['TLAT', 'TLON', 'hi', 'hs', 'tsnz', 'tair', 'snow', 'aice', 'iage']

    # Calculate averages per group (not aggregating full profiles)
    grouped = df.groupby(group_cols).agg({
        'tinz': lambda x: x.head(layers_ice).mean() if len(x) >= layers_ice else np.nan,
        'sinz': lambda x: x.head(layers_ice).mean() if len(x) >= layers_ice else np.nan}).reset_index()

    # Rename for clarity
    grouped = grouped.rename(columns={
        'tinz': 'temperature_average',
        'sinz': 'salinity_average'})

    return grouped

def remove_nonphysical(df, layers_ice=None):
    dropped_rows = set()

    # Surface snow temperature above melting point
    tsnz_violation = df[df['tsnz'] > 273.15].index
    dropped_rows.update(tsnz_violation)

    # Temperature value check
    temp_violation = df[df['temperature_average'] > 273.15].index
    dropped_rows.update(temp_violation)

    # Salinity value check
    sal_violation = df[df['salinity_average'] < 0.002].index
    dropped_rows.update(sal_violation)

    # Ice thickness nonphysical: hi <= 0
    hi_violation = df[df['hi'] <= 0].index
    dropped_rows.update(hi_violation)

    # Snow thickness nonphysical: hs < 0
    hs_violation = df[df['hs'] < 0].index
    dropped_rows.update(hs_violation)

    # Snow surface temperature too low: tsnz < 100K
    tsnz_low_violation = df[df['tsnz'] < 100].index
    dropped_rows.update(tsnz_low_violation)

    # Air temperature too low: tair < 100K
    tair_low_violation = df[df['tair'] < 100].index
    dropped_rows.update(tair_low_violation)

    # Drop invalid rows
    df = df.drop(index=dropped_rows).reset_index(drop=True)

    # Create OW_df only from aice < 0.15
    OW_df = df[df['aice'] < 0.15].copy().reset_index(drop=True)

    # Remove OW from main dataframe
    df = df[df['aice'] >= 0.15].reset_index(drop=True)

    return df, OW_df

def SIC_filter(minimum, maximum, df):
    # Create a boolean mask for valid rows (inside the range)
    valid_mask = (df['aice'] >= minimum) & (df['aice'] <= maximum)
    
    # Return a new DataFrame with only valid rows
    df_filtered = df[valid_mask].copy()
    
    return df_filtered




