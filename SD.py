import matplotlib.pyplot as plt 
from data_visualization import dtu_coolwarm_cmap, dtu_grey, dtu_blues_cmap,dtu_reds_cmap, dtu_navy
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

#%%
import numpy as np
# Load datasets
ds_365H = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_36.5_H_CIMR.nc")
ds_365V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_36.5_V_CIMR.nc")
ds_187V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_18.7_V_CIMR.nc")
ds_69V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_6.9_V_CIMR.nc")
print("Footprints loaded...")

# Extract tb arrays
tb37h = ds_365H["tb"].values
tb37v = ds_365V["tb"].values
tb18v = ds_187V["tb"].values
tb7v = ds_69V["tb"].values
print("Brightness temperatures extracted...")

lat = ds_365H["lat"].values
lon = ds_365H["lon"].values
print("Lat and lon extracted...")

# Compute SIC with Bristol
bristol_SIC_footprint = bristol_CIMR(tb18v, tb37v, tb37h)
print("Bristol SIC computed...")

# Acces CICE and merge with OW where SIC =0
CICE = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CICEv3.csv"
df_cice = pd.read_csv(CICE)
df_cice = df_cice.rename(columns={'TLAT': 'lat', 'TLON': 'lon'})

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


# Flatten brightness temperature arrays
flat_tb7v = tb7v.flatten()
flat_tb18v = tb18v.flatten()

# Create DataFrames for tb7v and tb18v
df_tb7v = pd.DataFrame({
    'lat': flat_lat,
    'lon': flat_lon,
    'tb7v': flat_tb7v
}).sort_values(by=['lat', 'lon']).reset_index(drop=True)

df_tb18v = pd.DataFrame({
    'lat': flat_lat,
    'lon': flat_lon,
    'tb18v': flat_tb18v
}).sort_values(by=['lat', 'lon']).reset_index(drop=True)

# Use same KDTree and indices to interpolate tb7v
tb7v_values = df_tb7v['tb7v'].values[indices]
weighted_tb7v = np.average(tb7v_values, axis=1, weights=weights)
df_cice['tb7v_nearest'] = weighted_tb7v

# Use same KDTree and indices to interpolate tb18v
tb18v_values = df_tb18v['tb18v'].values[indices]
weighted_tb18v = np.average(tb18v_values, axis=1, weights=weights)
df_cice['tb18v_nearest'] = weighted_tb18v



#%%

# %%
SIC = df_cice['aice'] 
# %%
tb7v = df_cice['tb7v_nearest']
tb19v =df_cice['tb18v_nearest']
iage = df_cice['iage']
SD_CICE= df_cice['hs']*100

import numpy as np

def snow_depth(tb7v, tb19v, sic, iage):
    """
    Compute snow depth on sea ice using brightness temperatures and sea ice concentration.

    Parameters:
    tb7v: Series or array of brightness temperatures at 6.9 GHz vertical
    tb19v: Series or array of brightness temperatures at 18.7 GHz vertical
    sic: Series or array of sea ice concentration (0-1)
    iage: Series or array of ice age indicator

    Returns:
    Snow depth in cm as a numpy array
    """
    # Tie-points for FYI from Ivanova et al.
    tp19v = CIMR["FYI"]["18.7V"]["tiepoint"]
    tp7v = CIMR["FYI"]["6.9V"]["tiepoint"]

    k1 = tp19v - tp7v
    k2 = tp19v + tp7v

    GR = (tb19v - tb7v - k1 * (1 - sic)) / (tb19v + tb7v - k2 * (1 - sic))

    # Vektoriseret valg af koefficienter afhængigt af isalder
    a = np.where(iage < 0.602, 19.26, 19.34)
    b = np.where(iage < 0.602, 553, 368)

    sd = a - b * GR

    return sd

SD = snow_depth(tb7v, tb19v, SIC, iage)


diff = SD - SD_CICE

def plot_diff(lat, lon, cvalue, colorbar_min, colorbar_max):
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                               central_latitude=90, central_longitude=0)})

    ax.set_extent([-180, 180, 68, 90], crs=ccrs.PlateCarree())

    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)

    # Add land with DTU grey
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face', facecolor=dtu_grey)
    ax.add_feature(land)
    ax.coastlines(resolution='50m', linewidth=0.5, color='black')
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor=dtu_navy)
    ax.add_feature(ocean, zorder=0)

    # Gridlines
   # Gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}
    
    # Disable the 0° meridian label
    gl.top_labels = True
    gl.bottom_labels = False
    gl.left_labels = True
    gl.right_labels = False
    

    # Scatter plot
    sc = ax.scatter(lon, lat,
                    c=cvalue,
                    cmap=dtu_coolwarm_cmap,
                    norm=norm,
                    s=0.1,
                    transform=ccrs.PlateCarree())

    # Colorbar below the plot, exclude 0 and move it closer
    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8)
    cbar.set_ticks([-20,20])
    cbar.set_label('Rostosky $h_s$ [cm]', fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=20)

    plt.show()

#%%

#%%
lat = df_cice['lat']
lon = df_cice['lon']
iage = df_cice['iage']  # sea ice age
colorbar_min = -20
colorbar_max = 20

# First filter based on iage > 0.602
iage_valid = iage > 0.602

# Apply iage filter
lat_iage_filtered = lat[iage_valid]
lon_iage_filtered = lon[iage_valid]
diff_iage_filtered = diff[iage_valid]

# Then remove extreme outliers in diff (only upper 1% removed)
lower, upper = diff_iage_filtered.quantile([0, 0.99])
outlier_valid = (diff_iage_filtered >= lower) & (diff_iage_filtered <= upper)

# Apply outlier filter
lat_filtered = lat_iage_filtered[outlier_valid]
lon_filtered = lon_iage_filtered[outlier_valid]
diff_filtered = diff_iage_filtered[outlier_valid]

# Plot with both filters
plot_diff(lat_filtered, lon_filtered, diff_filtered, colorbar_min, colorbar_max)

print(diff_filtered.describe())

