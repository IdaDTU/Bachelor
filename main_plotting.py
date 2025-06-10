# Load modules
from data_visualization import plot_laea_categorical, plot_laea_cmap, dtu_navy, dtu_reds_cmap, plot_regular, dtu_blues_cmap
from footprint_operator import make_kernel
from dictionaries import CIMR_tracks, MWI_tracks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#%% -------------------------- User Input -------------------------- # 

# CICE paths
cice_input_path = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CICEv3.csv"
cice_output_path = "C:/Users/user/OneDrive/Desktop/Bachelor/plots/CICE_NCAT.png"
cice_unfiltered_input_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/CICE_unfiltered"

# SMRT paths
input_path_365GHz_h = "C:/Users/user/OneDrive/Desktop/Bachelor/test/fyi_myi_365H"
OW_161 = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/OW_161.csv"
OW_145 = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/Tiepoints/OW_145.29.csv"


#%%
fy_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/combine/31.4_V_FY_MWI_combined.csv"
my_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/combine/31.4_V_MY_MWI_combined.csv"
combined  = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\combine\36.5_H_MY_CIMR_combined_test.csv"
combined_MWI = pd.read_csv(combined)
lat = combined_MWI['lat']
lon = combined_MWI['lon']
tb = combined_MWI['tb']

colorbar_min = 150
colorbar_max = 250
plot_laea_cmap(lat, lon, tb, colorbar_min, colorbar_max, filename='31.4V_MWI.png')


#%% ---------------------------------------------------------------- # 

# CICE plots
CICE_df = pd.read_csv(cice_input_path)

#%% Extract CICE data for plots
CICE_lat = CICE_df['TLAT']
CICE_lon = CICE_df['TLON']
CICE_hi = CICE_df['hi']
CICE_aice = CICE_df['aice']
colorbar_min = 0
colorbar_max = 160
plot_laea_cmap(CICE_lat, CICE_lon, CICE_aice, colorbar_min, colorbar_max, filename='test.png')

#%%
CICE_hs = CICE_df['hs']
print(CICE_hs.min())
#%%
# Call plotting functions
plot_laea_categorical(CICE_lat, CICE_lon, CICE_hi ,cice_output_path)


#%%  
# SMRT plots
# Load and merge two CSVs

ice = pd.read_csv("C:/Users/user/OneDrive/Desktop/Bachelor/CSV/combine/18.7_V_MWI")

#%%
SMRT_lat = ice['lat']
SMRT_lon = ice['lon']
SMRT_tb = ice['tb']
colorbar_min = 210
colorbar_max = 250
tiepoint = 250

#%%
plot_laea_cmap(SMRT_lat, SMRT_lon, SMRT_tb , colorbar_min, colorbar_max)
#%%

#%%

fig, axes = plt.subplots(2, max(len(MWI_tracks), len(CIMR_tracks)), figsize=(15, 8), sharex=True, sharey=True)

# --- MWI plots ---
for i, (freq, props) in enumerate(MWI_tracks.items()):
    sigma_cross = props["IFOV"]["cross_track"]/4
    sigma_along = props["IFOV"]["along_track"]/4

    kernel = make_kernel(sigma_cross, sigma_along)
    im_mwi = axes[0, i].imshow(kernel.T, cmap=dtu_blues_cmap, interpolation='none')

    axes[0, i].text(0.02, 0.02, f"{freq} GHz", fontsize=25,
                    transform=axes[0, i].transAxes, verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    axes[0, i].tick_params(labelsize=25)
    #axes[0, i].set_xticklabels([])

# Remove unused MWI axes
for j in range(len(MWI_tracks), max(len(MWI_tracks), len(CIMR_tracks))):
    fig.delaxes(axes[0, j])

# Colorbar for MWI
cax_mwi = fig.add_axes([0.91, 0.58, 0.02, 0.35])
cb_mwi = fig.colorbar(im_mwi, cax=cax_mwi, orientation='vertical')
cb_mwi.set_label('Intensity', fontsize=25)
cb_mwi.ax.tick_params(labelsize=25)

# --- CIMR plots ---
for i, (freq, props) in enumerate(CIMR_tracks.items()):
    sigma_cross = props["IFOV"]["cross_track"]/3
    sigma_along = props["IFOV"]["along_track"]/3

    kernel = make_kernel(sigma_cross, sigma_along)
    im_cimr = axes[1, i].imshow(kernel.T, cmap=dtu_reds_cmap, interpolation='none')

    axes[1, i].text(0.02, 0.02, f"{freq} GHz", fontsize=25,
                    transform=axes[1, i].transAxes, verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    axes[1, i].tick_params(labelsize=25)

# Remove unused CIMR axes
for j in range(len(CIMR_tracks), max(len(MWI_tracks), len(CIMR_tracks))):
    fig.delaxes(axes[1, j])

# Colorbar for CIMR
cax_cimr = fig.add_axes([0.91, 0.13, 0.02, 0.35])
cb_cimr = fig.colorbar(im_cimr, cax=cax_cimr, orientation='vertical')
cb_cimr.set_label('Intensity', fontsize=25)
cb_cimr.ax.tick_params(labelsize=25)

# Labels
fig.supxlabel("Along scan [pixels]", fontsize=25, y=0.05)
fig.supylabel("Cross scan [pixels]", fontsize=25, x=0.01)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()


#%%

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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

# Extract CICE data for plots
CICE_lat = CICE_df['TLAT']
CICE_lon = CICE_df['TLON']
CICE_hi = CICE_df['temperature_average']
CICE_aice = CICE_df['aice']
colorbar_min = 280
colorbar_max = 300

def plot_diff(lat, lon, cvalue, colorbar_min, colorbar_max):
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                               central_latitude=90, central_longitude=0)})

    ax.set_extent([-180, 180, 68, 90], crs=ccrs.PlateCarree())

    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)

    # Ocean background with DTU navy
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=dtu_navy)

    # Add land with DTU grey
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face', facecolor=dtu_grey)
    ax.add_feature(land)
    ax.coastlines(resolution='50m', linewidth=0.5, color='black')

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
                    cmap=dtu_blues_cmap,
                    norm=norm,
                    s=0.1,
                    transform=ccrs.PlateCarree())

    # Colorbar below the plot
    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8)
    cbar.set_ticks([280,300])
    cbar.set_label('DMI-CICE $h_s$ [cm]', fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=20)

    plt.show()

#%%plot_diff(CICE_lat, CICE_lon, CICE_aice, colorbar_min, colorbar_max)

plot_diff(CICE_lat, CICE_lon, CICE_hi, colorbar_min, colorbar_max)

#%%
# Load datasets
CICE_df = pd.read_csv("C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/CICEv3.csv")
OW_df = pd.read_csv("C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/OWv3")
#%%
print(OW_df['tsnz'].describe())
#%%
# Combine datasets
import pandas as pd

# Combine and drop duplicates
df = pd.concat([OW_df, CICE_df]).drop_duplicates(subset=['TLAT', 'TLON'], keep='last')

# Extract necessary data
lat = df['TLAT']
lon = df['TLON']
tsnz = df['tsnz']
hs = df['hs']
#%%
print(hs.max())
#%%

# Remove extreme outliers
lower, upper = tsnz.quantile([0.0, 1])
valid = (tsnz >= lower) & (tsnz <= upper)

# Filtered data
lat_filtered = lat[valid]
lon_filtered = lon[valid]
tsnz_filtered = tsnz[valid]

# Cap values at 273.15
tsnz_filtered = tsnz_filtered.clip(upper=273.15)

# Create a DataFrame for sorting
filtered_df = pd.DataFrame({
    'lat': lat_filtered,
    'lon': lon_filtered,
    'tsnz': tsnz_filtered
})

# Sort by latitude, then longitude
filtered_df = filtered_df.sort_values(by=['lat', 'lon'])

# Define colorbar range
colorbar_min = 250
colorbar_max = 273.15

# Plot
plot_diff(filtered_df['lat'], filtered_df['lon'], filtered_df['tsnz'], colorbar_min, colorbar_max)



#%%
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

# Example DTU navy and DTU grey colors
dtu_navy = '#003366'
dtu_grey = '#666666'


# Extract CICE data for plots
CICE_lat = CICE_df['TLAT']
CICE_lon = CICE_df['TLON']


def calculate_true_interface_temperature(thickness_snow, thickness_ice, temperature_air, temperature_water=271.35):
    """
    Calculate the temperature at the snow-ice interface (Tsi) assuming thermal equilibrium.

    Parameters:
    thickness_snow: Snow thickness in meters (float or array-like)
    thickness_ice: Ice thickness in meters (float or array-like)
    temperature_air: Air temperature in Kelvin (float or array-like)
    temperature_water: Water temperature in Kelvin (default: 271.35 K)

    Returns:
    Interface temperature in Kelvin
    """
    ki = 2.1  # Ice thermal conductivity (W/m/K)
    ks = 0.4  # Snow thermal conductivity (W/m/K)

    numerator = ki * thickness_snow * temperature_water + ks * thickness_ice * temperature_air
    denominator = ki * thickness_snow + ks * thickness_ice
    interface_temp = numerator / denominator

    return interface_temp

CICE_hi = CICE_df['tair']
#calculate_true_interface_temperature(CICE_df['hs'],CICE_df['hi'],CICE_df['tair'])

colorbar_min = 260
colorbar_max = 273

def plot_diff(lat, lon, cvalue, colorbar_min, colorbar_max):
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                               central_latitude=90, central_longitude=0)})

    ax.set_extent([-180, 180, 68, 90], crs=ccrs.PlateCarree())

    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)

    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=dtu_navy)

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
                    cmap=dtu_blues_cmap,
                    norm=norm,
                    s=100,
                    transform=ccrs.PlateCarree())

    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8)
    cbar.set_ticks([260, 273])
    cbar.set_label('DMI-CICE $h_s$ [cm]', fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=20)

    plt.show()

# Compute IQR and detect outliers
Q1 = np.percentile(CICE_hi, 25)
Q3 = np.percentile(CICE_hi, 75)
IQR = Q3 - Q1

# Outlier thresholds
lower_bound = Q1 - 0.5 * IQR
upper_bound = Q3 + 0.5* IQR

# Mask only outliers
outlier_mask = (CICE_hi < lower_bound) | (CICE_hi > upper_bound)

# Subset the data
outlier_lat = CICE_lat[outlier_mask]
outlier_lon = CICE_lon[outlier_mask]
outlier_hi = CICE_hi[outlier_mask]

# Calculate values
num_outliers = outlier_hi.size
total_points = CICE_hi.size
percent_outliers = (num_outliers / total_points) * 100
mean_outlier = np.mean(outlier_hi)
min_outlier = np.min(outlier_hi)
max_outlier = np.max(outlier_hi)

# Print out the calculated values
print(f"Number of outliers: {num_outliers}")
print(f"Total number of points: {total_points}")
print(f"Percent of outliers: {percent_outliers:.2f}%")
print(f"Mean of outlier values: {mean_outlier:.2f}")
print(f"Min outlier value: {min_outlier:.2f}")
print(f"Max outlier value: {max_outlier:.2f}")

# Plot only outliers
plot_diff(outlier_lat, outlier_lon, outlier_hi, colorbar_min, colorbar_max)

#%%

