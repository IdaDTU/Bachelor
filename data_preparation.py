import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPolygon

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
    It selects one representative row per unique location+ice state group based on the maximum 'aicen' value.
    Then it gathers temperature and salinity profiles (tinz, sinz) from all layers for each group.
    
    Parameters:
        ds (xarray.Dataset): Dataset with sea ice variables.
        layers_ice (int): Number of ice layers to include in the temperature and salinity profiles.
    
    Returns:
        pd.DataFrame: A DataFrame with one row per group including metadata and the temperature/salinity profiles.
    """
    
    # Keep only the relevant variables from the dataset
    ds = ds[['TLAT', 'TLON', 'hi', 'hs', 'Sinz', 'Tinz', 'Tsnz', 'Tair', 'snow', 'nc','aicen']]

    # Convert the xarray Dataset to a pandas DataFrame and reset the index
    df = ds.to_dataframe().reset_index()

    # Convert temperatures from Celsius to Kelvin and salinity from g/m³ to kg/m³
    df['tinz'] = df['Tinz'] + 273.15
    df['tsnz'] = df['Tsnz'] + 273.15
    df['tair'] = df['Tair'] + 273.15
    df['sinz'] = df['Sinz'] / 1000

    # Drop the original variables after conversion to avoid confusion
    df.drop(columns=['Tinz', 'Tsnz', 'Tair', 'Sinz'], inplace=True)

    # Remove rows with missing values and filter for Arctic latitudes
    df = df.dropna()
    df = df[df['TLAT'] >= 55.5]

    # Define the grouping columns based on spatial location and surface variables
    group_cols = ['TLAT', 'TLON', 'hi', 'hs', 'tsnz', 'tair', 'snow']

    # For each group, identify the row with the maximum 'aicen' value to serve as the representative row
    idx_max_aicen = df.groupby(group_cols)['aicen'].idxmax()
    df_max_aicen = df.loc[idx_max_aicen].reset_index(drop=True)

    # Aggregate temperature and salinity values for each group into lists representing vertical profiles
    profiles = ( df.groupby(group_cols).agg({ 'tinz': lambda x: list(x)[:layers_ice],  # Limit profile to the specified number of layers
                                             'sinz': lambda x: list(x)[:layers_ice]}).reset_index())

    # Merge the representative row information with the profile data
    merged_df = pd.merge(df_max_aicen, profiles, on=group_cols, suffixes=('', '_profile'))

    # Rename the profile columns for clarity
    merged_df = merged_df.rename(columns={
        'tinz_profile': 'temperature_profiles',
        'sinz_profile': 'salinity_profiles'})

    # Convert the profile lists to numpy arrays for consistency and potential numerical operations
    merged_df['temperature_profiles'] = merged_df['temperature_profiles'].apply(np.array)
    merged_df['salinity_profiles'] = merged_df['salinity_profiles'].apply(np.array)

    return merged_df

def check_land(df):
    # Helper function
    # Set up Basemap with North Polar Stereographic projection
    m = Basemap(projection='npstere',
                boundinglat=66.5,
                lon_0=0,
                resolution='l')
    
    # Collect land polygons into a MultiPolygon
    land_polygons = []
    for polygon in m.drawcoastlines(linewidth=0.1).get_paths():
        try:
            coords = polygon.vertices
            x, y = zip(*coords)
            lon, lat = m(x, y, inverse=True)
            poly_coords = list(zip(lon, lat))
            land_polygons.append(Polygon(poly_coords))
        except:
            continue
    land_mask = MultiPolygon(land_polygons)

    # Check if each point is on land
    def is_land(lat, lon):
        x, y = m(lon, lat)
        return land_mask.contains(Point(x, y))

    df['land'] = df.apply(lambda row: is_land(row['TLAT'], row['TLON']), axis=1)
    
    return df

def remove_nonphysical(df, layers_ice):
    # Remove land pixels using a helper function
    df = check_land(df)
    
    # Keep only ocean pixels (land == False)
    df = df[df['land'] == False]
    
    # Remove rows with surface snow temperature (tsnz) above melting point
    df = df[df['tsnz'] <= 273.15]
    
    # Set to collect indices of rows that violate physical constraints
    dropped_rows = set()

    # Check temperature profiles: all values in each profile must be below or equal to 273.15 K
    for i in range(len(df['temperature_profiles'])):
        for j in range(layers_ice):
            if df['temperature_profiles'].iloc[i][j] > 273.15:
                dropped_rows.add(df.index[i])
                break  # No need to check more layers for this profile

    # Check salinity profiles: all values in each profile must be >= 0.002 (avoid non-physical salinity)
    for i in range(len(df['salinity_profiles'])):
        for j in range(layers_ice):
            if df['salinity_profiles'].iloc[i][j] < 0.002:
                dropped_rows.add(df.index[i])
                break  # Skip the rest once a non-physical value is found

    # Drop rows that violated any of the above constraints
    df = df.drop(list(dropped_rows))

    return df


def SIC_filter(minimum, maximum, df):
    # Create a boolean mask for valid rows (inside the range)
    valid_mask = (df['aice'] >= minimum) & (df['aice'] <= maximum)
    
    # Return a new DataFrame with only valid rows
    df_filtered = df[valid_mask].copy()
    
    return df_filtered







