import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, Normalize,ListedColormap, BoundaryNorm
import matplotlib


matplotlib.rcParams['font.family'] = 'Arial'

# DTU color scheme
dtu_navy = '#030F4F'
dtu_red = '#990000'
dtu_grey = '#DADADA'
white = '#ffffff'
black = '#000000'

# Color intermediates
phase1_blue = '#030F4F'
phase2_blue = '#3d4677'
phase3_blue = '#babecf'
phase1_red = '#990000'
phase2_red = '#bc5959'
phase3_red = '#e6c1c1'

# Color lists for colormaps
dtu_coolwarm = [dtu_navy, white, dtu_red]
dtu_blues = [dtu_navy, white]
dtu_reds = [dtu_red, white]

# Custom colormaps
dtu_coolwarm_cmap = LinearSegmentedColormap.from_list("dtu_coolwarm", dtu_coolwarm)
dtu_blues_cmap = LinearSegmentedColormap.from_list("dtu_blues", dtu_blues)
dtu_reds_cmap = LinearSegmentedColormap.from_list("dtu_reds", dtu_reds)


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from pyproj import Proj


# Conversion function for LAEA projection (optional, useful for checks/calculations)
def latlon_to_laea(lat, lon, lat_0=90, lon_0=0):
    laea_proj = Proj(proj='laea', lat_0=lat_0, lon_0=lon_0, ellps='WGS84')
    return laea_proj(lon, lat)

# Main plotting function using pcolormesh
def plot_laea_cmap(grid_lat, grid_lon, grid_tb,
                         colorbar_min, colorbar_max,
                         filename='plot_pcolormesh.pdf'):

    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(central_latitude=90,
                                                                                    central_longitude=0)})

    ax.set_extent([-180, 180, 66.5, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=dtu_grey)
    ax.add_feature(cfeature.COASTLINE)

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}

    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)

    # Plot using pcolormesh
    mesh = ax.pcolormesh(grid_lon, grid_lat, grid_tb,
                         cmap=dtu_coolwarm_cmap,
                         shading='auto',
                         norm=norm,
                         transform=ccrs.PlateCarree())

    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8, pad=0.09)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('TbH [K]', fontsize=18, labelpad=15)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Example call (replace with your actual data):
# plot_laea_pcolormesh(grid_lat, grid_lon, grid_tb, 200, 300)
def plot_laea_categorical(lat, lon, cvalue, output_path):
    
    """
    Plot categorical data on a LAEA projection map.

    Parameters:
    lat: 1D array of latitudes
    lon: 1D array of longitudes
    cvalue: 1D array of values to categorize
    output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(central_latitude=90,
                                                                                    central_longitude=0)})

    bins = [-np.inf, 0, 0.15, 0.30, 0.70, 1.20, 2.0, np.inf]
    colors = [dtu_grey, '#003366', '#204d80', '#4d73b3', '#7aa1cc', '#a7c8e6', '#d3e6f5', '#ffffff']
    categories = ['Land', 'OW', '0-0.15', '0.15-0.30', '0.30-0.70', '0.70-1.20', '1.20-2.0', '2.0+']

    # Digitize into bins
    cvalue_binned = np.digitize(cvalue, bins, right=False) - 1

    ax.set_extent([-180, 180, 66.5, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor=dtu_navy)
    ax.add_feature(cfeature.LAND, facecolor=dtu_grey)
    ax.add_feature(cfeature.COASTLINE)

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(categories) + 0.5), cmap.N)

    # Scatter plot
    sc = ax.scatter(lon, lat, c=cvalue_binned,
                    cmap=cmap, norm=norm,
                    s=0.1, edgecolor='none',
                    transform=ccrs.PlateCarree())

    # Colorbar setup
    cbar = plt.colorbar(sc, ax=ax, fraction=0.044, pad=0.09,
                        boundaries=np.arange(-0.5, len(categories) + 0.5))
    cbar.set_ticks(np.arange(len(categories)))
    cbar.set_ticklabels(categories)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Ice Thickness Category [m]', fontsize=18, labelpad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()




def plot_regular(lat, lon, cvalue):
    plt.figure(figsize=(8, 6))
    plt.scatter(lon, lat, c=cvalue, cmap='viridis', s=10)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
