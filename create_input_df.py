import os
import xarray as xr
import pandas as pd

def combine_nc_files(directory):
    """
    Open and combine all netCDF files in the given directory into a single dataset.

    Parameters:
        directory (str): Path to the directory containing the .nc files.

    Returns:
        xarray.Dataset: Combined dataset along the time dimension.
    """
    try:
        # List all .nc files in the directory
        nc_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".nc")]

        if not nc_files:
            raise FileNotFoundError("No .nc files found in the directory.")

        # Open and concatenate datasets manually
        datasets = [xr.open_dataset(file, decode_times=False) for file in nc_files]
        combined_ds = xr.concat(datasets, dim="time")  # Time dimmension 

        return combined_ds
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def create_input_dataframe(ds, 
                           time=0,
                           nc=5,
                           nkice=0,
                           nksnow=0, 
                           nj=1400,
                           ni=slice(None)):
    """
    Create a Pandas DataFrame from specific variables in a given dataset with configurable slicing.
    
    Parameters:
    ds (dataset): A dataset containing multiple variables.
    time (int): Index for the time dimension.
    nc (int): Index for the nc dimension.
    nkice (int): Index for the nkice dimension.
    nksnow (int): Index for the nksnow dimension.
    nj (int): Index for the nj dimension.
    ni (int or slice): Index or range for the ni dimension.
    
    Returns:
    DataFrame: Cleaned DataFrame with selected variables and no missing values.
    """
    df = pd.DataFrame({
        # Variables with dimensions (time, nc, nkice, nj, ni)
        'tinz': ds['Tinz'][time, nc, nkice, nj, ni].values,
        'sinz': ds['Sinz'][time, nc, nkice, nj, ni].values,
        'tsnz': ds['Tsnz'][time, nc, nkice, nj, ni].values,
        
        # Variables with dimensions (time, nc, nj, ni)
        'vicen': ds['vicen'][time, nc, nj, ni].values,
        'vsnon': ds['vsnon'][time, nc, nj, ni].values,
        
        # Variables with dimensions (time, nj, ni)
        'tsfc': ds['Tsfc'][time, nj, ni].values,
        'aice': ds['aice'][time, nj, ni].values,
        'hi': ds['hi'][time, nj, ni].values,
        'hs': ds['hs'][time, nj, ni].values,
        'sst': ds['sst'][time, nj, ni].values,
        'sss': ds['sss'][time, nj, ni].values,
        'uvel': ds['uvel'][time, nj, ni].values,
        'vvel': ds['vvel'][time, nj, ni].values,
        'uatm': ds['uatm'][time, nj, ni].values,
        'vatm': ds['vatm'][time, nj, ni].values,
        'sice': ds['sice'][time, nj, ni].values,
        'fswdn': ds['fswdn'][time, nj, ni].values,
        'flwdn': ds['flwdn'][time, nj, ni].values,
        'snow': ds['snow'][time, nj, ni].values,
        'uocn': ds['uocn'][time, nj, ni].values,
        'vocn': ds['vocn'][time, nj, ni].values,
        'frzmlt': ds['frzmlt'][time, nj, ni].values,
        'scale_factor': ds['scale_factor'][time, nj, ni].values,
        'fswint_ai': ds['fswint_ai'][time, nj, ni].values,
        'fswabs': ds['fswabs'][time, nj, ni].values,
        'albsni': ds['albsni'][time, nj, ni].values,
        'albice': ds['albice'][time, nj, ni].values,
        'albsno': ds['albsno'][time, nj, ni].values,
        'albpnd': ds['albpnd'][time, nj, ni].values,
        'flat': ds['flat'][time, nj, ni].values,
        'fsens': ds['fsens'][time, nj, ni].values,
        'evap': ds['evap'][time, nj, ni].values,
        'tair': ds['Tair'][time, nj, ni].values,
        'snoice': ds['snoice'][time, nj, ni].values,
        'fbot': ds['fbot'][time, nj, ni].values,
        'fhocn': ds['fhocn'][time, nj, ni].values,
        'fswthru': ds['fswthru'][time, nj, ni].values,
        'strairx': ds['strairx'][time, nj, ni].values,
        'strairy': ds['strairy'][time, nj, ni].values,
        'strtltx': ds['strtltx'][time, nj, ni].values,
        'strtlty': ds['strtlty'][time, nj, ni].values,
        'strcorx': ds['strcorx'][time, nj, ni].values,
        'strcory': ds['strcory'][time, nj, ni].values,
        'strocnx': ds['strocnx'][time, nj, ni].values,
        'strocny': ds['strocny'][time, nj, ni].values,
        'strintx': ds['strintx'][time, nj, ni].values,
        'strinty': ds['strinty'][time, nj, ni].values,
        'strength': ds['strength'][time, nj, ni].values,
        'divu': ds['divu'][time, nj, ni].values,
        'shear': ds['shear'][time, nj, ni].values,
        'iage': ds['iage'][time, nj, ni].values,
        'fyarea': ds['FYarea'][time, nj, ni].values,
        'apond': ds['apond'][time, nj, ni].values,
        'hpond': ds['hpond'][time, nj, ni].values,
        'ipand': ds['ipond'][time, nj, ni].values,
        'apeff': ds['apeff'][time, nj, ni].values,
        
        # Variables with dimensions (nj, ni)???
        'tarea': ds['tarea'][0, 0].values,
        'tmask': ds['tmask'][0, 0].values,
        'TLAT': ds['TLAT'][0, 0].values,
        'TLON': ds['TLON'][0, 0].values,
        'ULAT': ds['ULAT'][0, 0].values,
        'ULON': ds['ULON'][0, 0].values,
        'NLON': ds['NLON'][0, 0].values,
        'NLAT': ds['NLAT'][0, 0].values,
        'ELON': ds['ELON'][0, 0].values,
        'ELAT': ds['ELAT'][0, 0].values,
        
        # Variables with dimension (time)
        'time': ds['time'][time].values,
        
        # Variables with dimension (nkice)
        'vgrdi': ds['VGRDi'][nkice].values,
        
        # Variables with dimension (nksnow)
        'vgrds': ds['VGRDs'][nksnow].values,
        
        # Variables with dimension (nc)
        'ncat': ds['NCAT'][nc].values
    })

    return df.dropna()

