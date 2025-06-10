import matplotlib.pyplot as plt 
from data_visualization import dtu_coolwarm_cmap, dtu_grey, dtu_blues_cmap,dtu_reds_cmap
from dictionaries import CIMR
from algorhitms import bristol,bristol_CIMR
from skimage.transform import downscale_local_mean
from footprint_operator import resample
import pandas as pd
import xarray as xr
from dictionaries import MWI,OW_tiepoints
import numpy as np

# Load datasets
ds_314H = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_36.5_H_CIMR.nc")
ds_314V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_36.5_V_CIMR.nc")
ds_187V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_18.7_V_CIMR.nc")
print("Footprints loaded...")

# Extract tb arrays
tb37h = ds_314H["tb"].values
tb37v = ds_314V["tb"].values
tb18v = ds_187V["tb"].values
print("Brightness temperatures extracted...")

lat = ds_314H["lat"].values
lon = ds_314H["lon"].values
print("Lat and lon extracted...")

# Compute SIC with Bristol
bristol_SIC_footprint = bristol_CIMR(tb18v, tb37v, tb37h)
print("Bristol SIC computed...")

# Acces CICE and merge with OW where SIC =0
CICE = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\CICE\CICE"
OW = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\CICE\OW_0_SIC.csv"
df_cice = pd.read_csv(CICE)
df_OW = pd.read_csv(OW)

df_cice = df_cice.rename(columns={'TLAT': 'lat', 'TLON': 'lon'})
df_OW = df_OW.rename(columns={'TLAT': 'lat', 'TLON': 'lon'})




#%%


# Flatten arrays
flat_lat = lat.flatten()
flat_lon = lon.flatten()
flat_sic = bristol_SIC_footprint.flatten()


# Create DataFrame
df_sic = pd.DataFrame({'lat': flat_lat,
                       'lon': flat_lon,
                       'bristol_sic': flat_sic})

df_sic = df_sic.sort_values(by=['lat', 'lon']).reset_index(drop=True)
print(df_sic)



#%%
import numpy as np
from scipy.spatial import cKDTree

def latlon_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)

# Convert coordinates to Cartesian
sic_cart = latlon_to_cartesian(df_sic['lat'].values, df_sic['lon'].values)
cice_cart = latlon_to_cartesian(df_cice['lat'].values, df_cice['lon'].values)

# Build KDTree
tree = cKDTree(sic_cart)

# Query 16 nearest neighbors
distances, indices = tree.query(cice_cart, k=16)

# Compute distance-weighted average
weights = 1 / (distances + 1e-12)  # avoid division by zero
weighted_sic = np.average(df_sic['bristol_sic'].values[indices], axis=1, weights=weights)

# Assign to df_cice
df_cice['bristol_sic_nearest'] = weighted_sic


    
#%%

# Flatten lat/lon
flat_lat = df_cice['lat']
flat_lon = df_cice['lon']

# Construct DataFrame
df_grid = pd.DataFrame({
    'lat': flat_lat,
    'lon': flat_lon,
    'bristol': df_cice['bristol_sic_nearest'],
    'aice': df_cice['aice']
})

# Sort
df_grid_sorted = df_grid.sort_values(by=['lat', 'lon']).reset_index(drop=True)



#%%

# Load SMRT CSVs
path_187V = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\SMRT\18.7_V_CIMR.csv"
path_314H = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\SMRT\36.5_H_CIMR.csv"
path_314V = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\SMRT\36.5_V_CIMR.csv"

df_187V = pd.read_csv(path_187V)
df_314H = pd.read_csv(path_314H)
df_314V = pd.read_csv(path_314V)

lat_smrt = df_187V['lat'].values
lon_smrt = df_187V['lon'].values
tb18v = df_187V['tb'].values
tb37h = df_314H['tb'].values
tb37v = df_314V['tb'].values

# Compute Bristol SIC
sic_smrt = bristol_CIMR(tb18v, tb37v, tb37h)

# Load CICE data
df_cice = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\CICE\CICE")
df_cice = df_cice.rename(columns={'TLAT': 'lat', 'TLON': 'lon'})


# Convert lat/lon to Cartesian for nearest-neighbor interpolation
def latlon_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)

smrt_cart = latlon_to_cartesian(lat_smrt, lon_smrt)
cice_cart = latlon_to_cartesian(df_cice['lat'].values, df_cice['lon'].values)

# KDTree interpolation (16 nearest neighbors with inverse distance weighting)
tree = cKDTree(smrt_cart)
distances, indices = tree.query(cice_cart, k=16)
weights = 1 / (distances + 1e-12)
interpolated_sic = np.average(sic_smrt[indices], axis=1, weights=weights)

# Build DataFrame for correlation and plotting
df_grid2 = pd.DataFrame({
    'lat': df_cice['lat'],
    'lon': df_cice['lon'],
    'bristol': interpolated_sic,
    'aice': df_cice['aice']
})




# Recalculate correlation
correlation_sorted = df_grid_sorted['bristol'].corr(df_grid2['bristol'])
print(f"Correlation between Bristol SIC and CICE aice (sorted grid): {correlation_sorted:.3f}")

