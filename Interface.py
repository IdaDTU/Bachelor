import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from algorhitms import bristol_CIMR
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

#%% Load datasets
ds_69V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_6.9_V_CIMR.nc")
print("Footprints loaded...")

# Extract tb arrays
tb69v = ds_69V["tb"].values
print("Brightness temperatures extracted...")

# Extract coordinates
lat = ds_69V["lat"].values
lon = ds_69V["lon"].values
print("Lat and lon extracted...")

# Access and prepare CICE data
CICE = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CICEv3.csv"
df_cice = pd.read_csv(CICE)
df_cice = df_cice.rename(columns={'TLAT': 'lat', 'TLON': 'lon'})

# Flatten arrays
flat_lat = lat.flatten()
flat_lon = lon.flatten()
flat_tb69v = tb69v.flatten()
print("Arrays flattened..")

# Create DataFrame for tb69v
df_tb69v = pd.DataFrame({'lat': flat_lat, 'lon': flat_lon, 'tb69v': flat_tb69v})
print("df created...")
# Sort for consistency
df_tb69v = df_tb69v.sort_values(by=['lat', 'lon']).reset_index(drop=True)

# Convert lat/lon to Cartesian coordinates
def latlon_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)

sic_cart = latlon_to_cartesian(df_tb69v['lat'].values, df_tb69v['lon'].values)
cice_cart = latlon_to_cartesian(df_cice['lat'].values, df_cice['lon'].values)

# Build KDTree and query nearest neighbors
tree = cKDTree(sic_cart)
distances, indices = tree.query(cice_cart, k=16)
weights = 1 / (distances + 1e-12)  # avoid division by zero

#%% Interpolate SIC
weighted_sic = np.average(df_tb69v['tb69v'].values[indices], axis=1, weights=weights)
df_cice['tb'] = weighted_sic


print("Interpolation of SIC and tb69v complete.")

import numpy as np

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

def compute_interface_temperature(TB_6V, Ti, SIC):
    """
    Compute ice temperature using the AMSR-E algorithm, only for SIC >= 0.90.

    Parameters:
    TB_6V: Brightness temperature at 6 GHz vertical polarization (float or array)
    Ti: True interface temperature (float or array)
    SIC: Ice concentration (0 to 1, float or array)

    Returns:
    T_si: Estimated ice temperature in Kelvin where SIC >= 0.90, np.nan elsewhere
    """
    T_w = 271.35  # Water temperature in Kelvin
    
    eps6V = np.mean(TB_6V) / np.mean(Ti)  # Emissivity estimate
    print(f"Emissivity at 6V: {eps6V}")

    T_p = TB_6V / eps6V

    # Initialize with NaNs
    T_si = np.full_like(TB_6V, np.nan)

    # Only compute where SIC >= 0.90
    mask = SIC >= 0.90
    T_si[mask] = (T_p[mask] - T_w * (1 - SIC[mask])) / SIC[mask]

    return T_si


#%%

df_cice['Ti'] = df_cice['temperature_average']
TB_6V = df_cice['tb']
SIC = df_cice['aice']
thickness_snow = df_cice['hs']
thickness_ice = df_cice['hi']
temperature_air =df_cice['tair']
Ti = df_cice['Ti'] 

tsi_from_tb = compute_interface_temperature(TB_6V, Ti, SIC)
true_tsi = calculate_true_interface_temperature(thickness_snow, thickness_ice, temperature_air)
#%%
from matplotlib.patches import Patch
# Plotting function
df_cice['tsi_from_tb'] = tsi_from_tb
df_cice['true_tsi'] = true_tsi
df_cice['difference'] = df_cice['tsi_from_tb']  - df_cice['true_tsi']
#%%
print(df_cice['difference'].describe())

#%%
def plot_diff(lat, lon, cvalue, colorbar_min, colorbar_max, filename):
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                               central_latitude=90, central_longitude=0)})
    ax.set_extent([-180, 180, 68, 90], crs=ccrs.PlateCarree())
    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face', facecolor=dtu_grey)
    ax.add_feature(land)
    # Add ocean with DTU navy
    #ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
    #                                     edgecolor='face', facecolor=dtu_navy)
    #ax.add_feature(ocean, zorder=0)
    ax.coastlines(resolution='50m', linewidth=0.5, color='black')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}
    gl.top_labels = True
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = True
    sc = ax.scatter(lon, lat, c=cvalue, cmap=dtu_coolwarm_cmap, norm=norm, s=0.1, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8)
    cbar.set_ticks([-15,15])
    cbar.set_label(r'(Barber $T_{s\mid i}$ - DMI-CICE $T_{s\mid i}$) [K]', fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=20)
    # Create a legend manually for OW
    #legend_elements = [Patch(facecolor=dtu_navy, edgecolor='black', label='OW and SIC < 0.9')]
    #ax.legend(handles=legend_elements, loc='upper left', fontsize=20, frameon=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


plot_diff(df_cice['lat'], df_cice['lon'], df_cice['difference'],
          colorbar_min=-15, colorbar_max=15, filename='tsi')



import matplotlib.pyplot as plt

#%% Scatter plot of tsi_from_tb vs aice
import numpy as np
import pandas as pd
from scipy.stats import linregress, zscore
import matplotlib.pyplot as plt

# Example: make sure df_cice is loaded before this
# df_cice = pd.read_csv('your_file.csv')



# Clip 'aice' values between 0 and 1
df_cice['aice'] = np.clip(df_cice['aice'], 0, 1)

# Remove rows with NaNs in 'aice' or 'tsi_from_tb'
df_clean = df_cice[['aice', 'true_tsi']].dropna()

# Remove extreme outliers based on z-score
z_scores = np.abs(zscore(df_clean))
df_clean = df_clean[(z_scores < 5).all(axis=1)]

# Perform linear regression: regress aice vs tsi_from_tb
slope, intercept, r_value, p_value, std_err = linregress(df_clean['tsi_from_tb'], df_clean['aice'])
r_squared = r_value ** 2

# Create scatter plot
plt.figure(figsize=(10, 10))

# Scatter plot: x is tsi_from_tb, y is aice
plt.scatter(df_clean['tsi_from_tb'], df_clean['aice'], color=dtu_navy, alpha=0.4)

# Add linear regression line
x_vals = np.linspace(df_clean['tsi_from_tb'].min(), df_clean['tsi_from_tb'].max(), 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, color=dtu_grey, linestyle='-', label=f'Linear fit: y = {slope:.2f}x + {intercept:.2f} (R² = {r_squared:.3f})')

# Labels
plt.xlabel('Estimated Interface Temperature (Tsi from TB) [K]', fontsize=24)
plt.ylabel('Sea Ice Concentration (aice) [Unitless]', fontsize=24)

# Set tick sizes
plt.tick_params(axis='both', labelsize=22)

# Set x and y limits
plt.xlim(df_clean['true_tsi'].min(), df_clean['true_tsi'].max())
plt.ylim(0.9, 1.0)

# Grid and legend
plt.grid(True)
plt.legend(loc='lower right', fontsize=22)

# Save and show plot
plt.tight_layout()
plt.savefig('scatter_tsi_vs_aice_regression.png', dpi=300)
plt.show()
