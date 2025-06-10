import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
import pandas as pd
from data_visualization import dtu_blues_cmap, dtu_grey, dtu_coolwarm_cmap, dtu_navy
from dictionaries import CIMR

def plot(lat, lon, cvalue, colorbar_min, colorbar_max):
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                               central_latitude=90, central_longitude=0)})

    ax.set_extent([-180, 180, 68, 90], crs=ccrs.PlateCarree())

    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)

    # Change the ocean color to dtu_navy and set it to the far back with zorder=-1
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=dtu_navy, zorder=-1)
    ax.add_feature(ocean)

    # Land feature with existing color
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face', facecolor=dtu_grey)
    ax.add_feature(land)
    
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
    cbar.set_label(r'36.5 GHz T$_{B}$(H) [K]', fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=20)
    plt.show()

#%% Load the CSV correctly
CIMR_187 = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\SMRT\combine\newnew_31.4H.csv"
CIMR_187_df = pd.read_csv(CIMR_187)
print(CIMR_187_df.mean())
#%%
OW = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\OW\OW_36.5H.csv"
OW_df = pd.read_csv(OW)

# Merge with sea ice priority (overwrites OW if same lat/lon exists)
df = pd.concat([OW_df, CIMR_187_df]).drop_duplicates(subset=['lat', 'lon'], keep='last')

lat = df['lat']
lon = df['lon']
tb = df['tb']
colorbar_min = 145
colorbar_max = 215

plot(lat, lon, tb, colorbar_min, colorbar_max)


#%%

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

# Load the NetCDF file
footprint = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_36.5_H_CIMR.nc")

# Access variables from the NetCDF file
lat = footprint['lat'].values
lon = footprint['lon'].values
tb = footprint['tb'].values

# %%
# Define colorbar limits

colorbar_min = 145
colorbar_max = 215

# Custom plot function
def plot(lat, lon, cvalue, colorbar_min, colorbar_max):
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                               central_latitude=90, central_longitude=0)})

    ax.set_extent([-180, 180, 68, 90], crs=ccrs.PlateCarree())

    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)

    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face', facecolor=dtu_grey)  # change this color to 'dtu_grey' if needed
    ax.add_feature(land)
    ax.coastlines(resolution='50m', linewidth=0.5, color='black')
    
    # Change the ocean color to dtu_navy and set it to the far back with zorder=-1
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=dtu_navy, zorder=-1)
    ax.add_feature(ocean)

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}
    gl.top_labels = True
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = True

    sc = ax.scatter(lon, lat,
                    c=cvalue,
                    cmap=dtu_coolwarm_cmap,  # You can replace this with a custom colormap if needed
                    norm=norm,
                    s=0.1,
                    transform=ccrs.PlateCarree())

    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8)
    cbar.set_label(r'Smeared 36.5 GHz T$_{B}$(H) [K]', fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=20)
    plt.show()

# Call the plot function
plot(lat, lon, tb, colorbar_min, colorbar_max)



