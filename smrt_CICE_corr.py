# -*- coding: utf-8 -*-
"""
Created on Fri May 23 16:59:00 2025

@author: user
"""

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
import matplotlib.pyplot as plt 
from data_visualization import dtu_coolwarm_cmap, dtu_grey, dtu_blues_cmap,dtu_reds_cmap,dtu_navy
from dictionaries import CIMR
from algorhitms import bristol,bristol_CIMR
from skimage.transform import downscale_local_mean
from footprint_operator import resample
import pandas as pd
import xarray as xr
from dictionaries import MWI,OW_tiepoints
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
# Load datasets
ds_314H = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_31.4_H_MWI.nc")
ds_314V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_31.4_V_MWI.nc")
ds_187V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_18.7_V_MWI.nc")
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

# Acces CICE and merge with OW where SIC =0
CICE = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CICEv3.csv"
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
path_187V = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\MWI\MWI_18.7GHz_V_merged.csv"
path_314H = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\MWI\MWI_31.4GHz_H_merged.csv"
path_314V = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\MWI\MWI_31.4GHz_V_merged.csv"

df_187V = pd.read_csv(path_187V)
df_314H = pd.read_csv(path_314H)
df_314V = pd.read_csv(path_314V)

lat_smrt = df_187V['lat'].values
lon_smrt = df_187V['lon'].values
tb18v = df_187V['tb'].values
tb37h = df_314H['tb'].values
tb37v = df_314V['tb'].values

# Compute Bristol SIC
sic_smrt = bristol(tb18v, tb37v, tb37h)

# Load CICE data
df_cice = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CICEv3.csv")
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

#%%
diff_smear = df_grid['bristol'] - df_grid['aice']
diff_smrt = df_grid2['bristol'] - df_grid2['aice']
master_diff = diff_smear -diff_smrt
print(diff_smear.describe())

# Load CICE data
df_cice = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CICEv3.csv")
df_cice = df_cice.rename(columns={'TLAT': 'lat', 'TLON': 'lon'})


#%%

def plot_diff(lat, lon, cvalue, colorbar_min, colorbar_max):
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                               central_latitude=90, central_longitude=0)})

    ax.set_extent([-180, 180, 68, 90], crs=ccrs.PlateCarree())

    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)

    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face', facecolor=dtu_grey)
    ax.add_feature(land)
    # Change the ocean color to dtu_navy and set it to the far back with zorder=-1
    #ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=dtu_navy, zorder=-1)
    #ax.add_feature(ocean)
    ax.coastlines(resolution='50m', linewidth=0.5, color='black')

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}
    gl.top_labels = True
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = True

    sc = ax.scatter(lon, lat,
                    c=cvalue,
                    cmap=dtu_coolwarm_cmap,
                    norm=norm,
                    s=0.1,
                    transform=ccrs.PlateCarree())

    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8)
    cbar.set_ticks([-0.2, 0, 0.2])
    cbar.set_label('MWI Smear Contribution [Unitless]', fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=20)
    plt.show()


plot_diff(df_grid['lat'], df_grid['lon'], master_diff, colorbar_min=-0.2, colorbar_max=0.2)

#%% Uncertainty plot:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming df_grid already has lat, lon columns
df_grid['diff_smear'] = master_diff
df_grid['diff_smrt'] = diff_smrt

# Merge df_grid with df_cice on lat and lon
merged = pd.merge(df_cice, df_grid[['lat', 'lon', 'diff_smear', 'diff_smrt']], on=['lat', 'lon'], how='inner')

# Filter aice between 0 and 1
merged = merged[(merged['aice'] >= 0) & (merged['aice'] <= 1)]

# Create bins of size 0.02
bins = np.arange(0, 1.01, 0.02)  # 0 to 1 inclusive
labels = bins[:-1] + 0.005  # bin centers

# Bin the aice values
merged['aice_bin'] = pd.cut(merged['aice'], bins=bins, labels=labels, include_lowest=True)

# Group by aice_bin and calculate mean diff_smear and diff_smrt
average_diff = merged.groupby('aice_bin')[['diff_smear', 'diff_smrt']].mean().reset_index()

# Compute tie-point uncertainty
epsilon_water = 0.02  # 2% for water
epsilon_ice = 0.005   # 0.5% for ice

alpha = labels  # Bin centers from 0 to 1
tiepoint_uncertainty = np.sqrt((1 - alpha) ** 2 * epsilon_water ** 2 + alpha ** 2 * epsilon_ice ** 2)

# Smearing uncertainty: standard deviation per bin of diff_smear
smearing_uncertainty = merged.groupby('aice_bin')['diff_smear'].std().reset_index()

# Merge uncertainties
uncertainty_df = pd.DataFrame({
    'aice_bin': labels,
    'tiepoint_uncertainty': tiepoint_uncertainty
})
uncertainty_df = pd.merge(uncertainty_df, smearing_uncertainty, on='aice_bin')
uncertainty_df.rename(columns={'diff_smear': 'smearing_uncertainty'}, inplace=True)

# Total uncertainty
uncertainty_df['total_uncertainty'] = np.sqrt(uncertainty_df['tiepoint_uncertainty']**2 + uncertainty_df['smearing_uncertainty']**2)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(average_diff['aice_bin'], average_diff['diff_smear'], label='Average diff_smear')
plt.plot(average_diff['aice_bin'], average_diff['diff_smrt'], label='Average diff_smrt')
plt.plot(uncertainty_df['aice_bin'], uncertainty_df['tiepoint_uncertainty'], label='Tie-point uncertainty', linestyle='--', color='red')
plt.plot(uncertainty_df['aice_bin'], uncertainty_df['total_uncertainty'], label='Total uncertainty', linestyle='--', color='black')
plt.xlabel('AICE (binned, center value)')
plt.ylabel('Difference / Uncertainty')
plt.title('Average diff_smear, diff_smrt, Tie-point and Total Uncertainty per AICE bin')
plt.grid(True)
plt.xlim([0.14, 1])
plt.legend()
plt.show()

#%%
