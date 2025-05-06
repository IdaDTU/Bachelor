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


def plot_npstere_cmap(lat, lon, cvalue, colorbar_min, colorbar_max, filename='plot.pdf'):
    """
    Plot scatter data on a North Polar Stereographic map using a continuous colormap.

    Parameters:
    lat: 1D array of latitudes
    lon: 1D array of longitudes
    cvalue: 1D array of values to color
    colorbar_min: minimum colorbar value
    colorbar_max: maximum colorbar value
    filename: output filename
    """
    fig, ax = plt.subplots(figsize=(10, 10),
                            subplot_kw={'projection': ccrs.NorthPolarStereo()})

    ax.set_extent([-180, 180, 66.5, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=dtu_grey)
    ax.add_feature(cfeature.COASTLINE)

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}

    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)

    sc = ax.scatter(lon, lat,
                    c=cvalue,
                    cmap=dtu_coolwarm_cmap,
                    norm=norm,
                    s=0.01,
                    transform=ccrs.PlateCarree())

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.8, pad=0.09)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('TbH [K]', fontsize=18, labelpad=15)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()




def plot_npstere_categorical(lat, lon, cvalue, output_path):
    """
    Plot scatter data on a North Polar Stereographic map with categorical colors.

    Parameters:
    lat: 1D array of latitudes
    lon: 1D array of longitudes
    cvalue: 1D array of continuous values to bin into categories
    output_path: Path to save the output figure
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.NorthPolarStereo()})

    # Define colors and categories
    bins = [-np.inf, 0, 0.15, 0.30, 0.70, 1.20, 2.0, np.inf]
    colors = [dtu_grey,'#003366', '#204d80', '#4d73b3', '#7aa1cc', '#a7c8e6', '#d3e6f5', '#ffffff']
    categories = ['Land','OW', '0-0.15', '0.15-0.30', '0.30-0.70', '0.70-1.20', '1.20-2.0', '2.0+']

    # Digitize values into bin indices (0-based)
    cvalue_binned = np.digitize(cvalue, bins, right=False) - 1

    # Map extent and features
    ax.set_extent([-180, 180, 66.5, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor=dtu_navy)
    ax.add_feature(cfeature.LAND, facecolor=dtu_grey)
    ax.add_feature(cfeature.COASTLINE)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}

    # Colormap and normalization
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(categories) + 0.5, 1), cmap.N)

    # Scatter plot
    sc = ax.scatter(lon, lat,
                    c=cvalue_binned,
                    cmap=cmap,
                    norm=norm,
                    s=0.1,
                    edgecolor='none',
                    transform=ccrs.PlateCarree())

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.044, pad=0.09,
                        boundaries=np.arange(-0.5, len(categories) + 0.5, 1))
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
