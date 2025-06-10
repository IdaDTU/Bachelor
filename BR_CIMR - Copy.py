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

# %%Load datasets
ds_314H = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_36.5_H_CIMR.nc")
ds_314V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_36.5_V_CIMR.nc")
ds_187V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_18.7_V_CIMR.nc")
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
CICE = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CICEv3.csv"
OW = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\OWv3.2"
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
print(df_sic['lat'].min(), df_sic['lat'].max())
print(df_cice['lat'].min(), df_cice['lat'].max())

#%%

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

import numpy as np

def plot_diff(lat, lon, cvalue, colorbar_min, colorbar_max, filename):
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                               central_latitude=90, central_longitude=0)})

    ax.set_extent([-180, 180, 68, 90], crs=ccrs.PlateCarree())

    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)

    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face', facecolor=dtu_grey)
    ax.add_feature(land)
    # Change the ocean color to dtu_navy and set it to the far back with zorder=-1
    # ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=dtu_navy, zorder=-1)
    # ax.add_feature(ocean)
    ax.coastlines(resolution='50m', linewidth=0.5, color='black')

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}
    gl.top_labels = True
    gl.bottom_labels = False
    gl.left_labels = True
    gl.right_labels = False

    sc = ax.scatter(lon, lat,
                    c=cvalue,
                    cmap=dtu_coolwarm_cmap,
                    norm=norm,
                    s=0.1,
                    transform=ccrs.PlateCarree())

    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8)
    cbar.set_ticks([-0.2, 0, 0.2])
    cbar.set_label('(CIMR Bristol SIC - DMI-CICE SIC) [Unitless]', fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


    
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

# Recalculate correlation
correlation_sorted = df_grid_sorted['bristol'].corr(df_grid_sorted['aice'])
print(f"Correlation between Bristol SIC and CICE aice (sorted grid): {correlation_sorted:.3f}")

# Plot the diff
diff = df_grid['bristol'] - df_grid['aice']


# Plot with highlight
plot_diff(df_grid['lat'], df_grid['lon'], diff,
                         colorbar_min=-0.2, colorbar_max=0.2,
                         filename='plot_with_sic015_overlay.png')




#%%

print(df_grid['bristol'].max())
print(df_cice['aice'].max())


