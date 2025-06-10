import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from data_visualization import dtu_coolwarm_cmap, dtu_grey, dtu_blues_cmap, dtu_reds_cmap, dtu_navy

# Load the data
CICE_df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CICEv3.csv")
OW_df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\OWv3.2")

# Merge the two DataFrames
merged_df = pd.concat([CICE_df, OW_df])

# Extract relevant columns
thickness_snow = merged_df['hs']
thickness_ice = merged_df['hi']
temperature_air = merged_df['tair']
latitudes = merged_df['TLAT']
longitudes = merged_df['TLON']

# Constants
ksi = 2.1
ks = 0.4
temperature_water = 271.35

# Interface temperature calculation
interface_temp = (ksi * thickness_snow * temperature_water + ks * thickness_ice * temperature_air) / \
                 (ksi * thickness_snow + ks * thickness_ice)

# Filter: only snow depth > 0.05 m
valid = thickness_snow > 0.05
thickness_snow_valid = thickness_snow[valid]
interface_temp_valid = interface_temp[valid]

# Linear regression (x: interface_temp, y: thickness_snow)
coefficients = np.polyfit(interface_temp_valid, thickness_snow_valid, 1)
regression_line = np.poly1d(coefficients)

# Scatter plot with regression line
plt.figure()
plt.scatter(interface_temp_valid, thickness_snow_valid, s=1)
x_vals = np.linspace(interface_temp_valid.min(), interface_temp_valid.max(), 100)
plt.plot(x_vals, regression_line(x_vals), color='red')
plt.xlabel('Interface Temperature (K)')
plt.ylabel('Snow Thickness (m)')
plt.title('Regression: Interface Temperature vs. Snow Thickness')
plt.show()

def plot_diff(lat, lon, cvalue, colorbar_min, colorbar_max):
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                               central_latitude=90, central_longitude=0)})

    ax.set_extent([-180, 180, 68, 90], crs=ccrs.PlateCarree())

    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)

    # Add ocean with DTU navy
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
                                         edgecolor='face', facecolor=dtu_navy)
    ax.add_feature(ocean, zorder=0)

    # Add land with DTU grey
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face', facecolor=dtu_grey)
    ax.add_feature(land, zorder=1)

    # Coastlines
    ax.coastlines(resolution='50m', linewidth=0.5, color='black')

    # Gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}
    gl.top_labels = True
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = True

    # Scatter plot
    sc = ax.scatter(lon, lat,
                    c=cvalue,
                    cmap=dtu_reds_cmap,
                    norm=norm,
                    s=0.1,
                    transform=ccrs.PlateCarree())

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8)
    cbar.set_ticks([250, 270])
    cbar.set_label('DMI-CICE $T_{s\\mid i}$ [K]', fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=20)

    # Create a legend manually for OW
    legend_elements = [Patch(facecolor=dtu_navy, edgecolor='black', label='OW')]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=20, frameon=True)

    plt.show()

plot_diff(latitudes, longitudes, interface_temp, colorbar_min=250, colorbar_max=270)
