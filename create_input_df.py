import os
import glob
import xarray as xr

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

def create_input_dataframe(ds):
    """
    Create a Pandas DataFrame from all variables in the dataset
    without applying any manual indexing. All dimensions (e.g., time,
    spatial coordinates) will become columns in the DataFrame.
    
    Rows where TLAT is below the Arctic Circle (i.e., TLAT < 66.5°) are dropped.
    Optionally, you can also filter by TLON if required.
    
    Parameters:
        ds (xarray.Dataset): The dataset containing multiple variables.
    
    Returns:
        pd.DataFrame: A DataFrame with all variables and coordinate labels,
                      with rows corresponding to every combination of coordinates.
    """
    
    # Select only the desired variables from the dataset.
    ds = ds[['TLAT', 'TLON', 'sst', 'hi','hs','Tair','Tinz']]

    # Convert the subset to a DataFrame and reset the index so that coordinates become columns.
    df = ds.to_dataframe().reset_index()
    
    # Convert Temperatures to Kelvin
    df['sst'] = df['sst'] + 273.15
    df['tair'] = df['Tair'] + 273.15
    df['tinz'] = df['Tinz'] + 273.15
    
    # Filter rows: Keep only rows where TLAT is within the Arctic Circle.
    df = df[df['TLAT'] >= 67]
    
    # Drop any rows with missing values.
    return df.dropna()
