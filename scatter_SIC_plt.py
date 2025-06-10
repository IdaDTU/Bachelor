import matplotlib.pyplot as plt 
from data_visualization import dtu_coolwarm_cmap, dtu_grey, dtu_blues_cmap,dtu_reds_cmap,dtu_red, dtu_navy
from dictionaries import CIMR
from algorhitms import bristol,bristol_CIMR
from skimage.transform import downscale_local_mean
from footprint_operator import resample
import pandas as pd
import xarray as xr
from dictionaries import MWI,OW_tiepoints
import numpy as np

#%%

# CICE Data
CICE = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\CICE\CICE"
OW = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\CICE\OW_0_SIC.csv"
df_cice = pd.read_csv(CICE)
df_OW = pd.read_csv(OW)

df_cice = df_cice.rename(columns={'TLAT': 'lat', 'TLON': 'lon'})
df_OW = df_OW.rename(columns={'TLAT': 'lat', 'TLON': 'lon'})


# Bristol data
# Load datasets
ds_314H = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_31.4_H_MWI.nc")
ds_314V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_31.4_V_MWI.nc")
ds_187V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_18.7_V_MWI.nc")
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
bristol_SIC_footprint = bristol(tb18v, tb37v, tb37h)
print("Bristol SIC computed...")

# Flatten arrays
flat_lat = lat.flatten()
flat_lon = lon.flatten()
flat_sic = bristol_SIC_footprint.flatten()

# Create DataFrame
df_sic = pd.DataFrame({'lat': flat_lat,
                       'lon': flat_lon,
                       'bristol_sic': flat_sic})

df_sic = df_sic.sort_values(by=['lat', 'lon']).reset_index(drop=True)


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
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Assuming df_cice is your dataframe with 'aice' and 'bristol_sic_nearest'

# Clip values of 'aice' and 'bristol_sic_nearest' to be between 0 and 1
df_cice['aice'] = np.clip(df_cice['aice'], 0, 1)
df_cice['bristol_sic_nearest'] = np.clip(df_cice['bristol_sic_nearest'], 0, 1)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(df_cice['aice'], df_cice['bristol_sic_nearest'])

# Calculate R-squared value
r_squared = r_value ** 2

# Create scatter plot
plt.figure(figsize=(10, 10))

# Scatter plot
plt.scatter(df_cice['aice'], df_cice['bristol_sic_nearest'], color=dtu_navy, alpha=0.4)

# Add the linear regression line (solid line)
plt.plot(df_cice['aice'], slope * df_cice['aice'] + intercept, color=dtu_grey, linestyle='-', label=f'Linear fit: y = {slope:.2f}x + {intercept:.2f} (R² = {r_squared:.3f})')

# Label the axes
plt.xlabel('DMI-CICE SIC [Unitless]', fontsize=24)
plt.ylabel('BR SIC [Unitless]', fontsize=24)

# Set tick sizes for x and y axes
plt.tick_params(axis='both', labelsize=22)

# Set x and y limits
plt.xlim(0.15, 1)
plt.ylim(0.15, 1)


# Show grid and legend with increased font size for the label
plt.grid(True)
plt.legend(loc='lower right', fontsize=22)

# Display the plot
plt.show()


