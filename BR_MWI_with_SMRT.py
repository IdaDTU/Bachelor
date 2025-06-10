import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree

from data_visualization import dtu_grey, dtu_reds_cmap,dtu_blues_cmap
from algorhitms import bristol_CIMR, bristol

# Load new SMRT MWI CSVs
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

# KDTree interpolation
tree = cKDTree(smrt_cart)
distances, indices = tree.query(cice_cart, k=16)
weights = 1 / (distances + 1e-12)
interpolated_sic = np.average(sic_smrt[indices], axis=1, weights=weights)

# Build DataFrame for correlation and plotting
df_grid = pd.DataFrame({
    'lat': df_cice['lat'],
    'lon': df_cice['lon'],
    'bristol': interpolated_sic,
    'aice': df_cice['aice']
})
df_grid_sorted = df_grid.sort_values(by=['lat', 'lon']).reset_index(drop=True)

# Correlation
correlation_sorted = df_grid_sorted['bristol'].corr(df_grid_sorted['aice'])
print(f"Correlation between SMRT Bristol SIC (MWI) and CICE aice: {correlation_sorted:.3f}")

# Plotting function
def plot_diff(lat, lon, cvalue, colorbar_min, colorbar_max, filename):
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                               central_latitude=90, central_longitude=0)})
    ax.set_extent([-180, 180, 66.5, 90], crs=ccrs.PlateCarree())
    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)
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
    sc = ax.scatter(lon, lat, c=cvalue, cmap=dtu_blues_cmap, norm=norm, s=0.1, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8)
    cbar.set_ticks([0, 0.1, 0.2])
    cbar.set_label('ΔSIC [Unitless]', fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Compute and plot difference
diff = df_grid_sorted['bristol'] - df_grid_sorted['aice']
plot_diff(df_grid_sorted['lat'], df_grid_sorted['lon'], diff,
          colorbar_min=0, colorbar_max=0.2, filename='smrt_mwi_vs_cice_diff.png')
