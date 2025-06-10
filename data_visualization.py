import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, Normalize,ListedColormap, BoundaryNorm
import matplotlib
from pyproj import Proj
from scipy.interpolate import griddata

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
dtu_blues = ['#030f4f',white]
dtu_reds = [white,dtu_red]

# Custom colormaps
dtu_coolwarm_cmap = LinearSegmentedColormap.from_list("dtu_coolwarm", dtu_coolwarm)
dtu_blues_cmap = LinearSegmentedColormap.from_list("dtu_blues", dtu_blues)
dtu_reds_cmap = LinearSegmentedColormap.from_list("dtu_reds", dtu_reds)

# LAEA projection helper
def latlon_to_laea(lat, lon, lat_0=90, lon_0=0):
    laea_proj = Proj(proj='laea', lat_0=lat_0, lon_0=lon_0)
    return laea_proj(lon, lat)  # pyproj uses (lon, lat)


def plot_laea_cmap(lat, lon, cvalue, colorbar_min, colorbar_max, filename='plot.pdf'):
    """
    Plot scatter data on a North Polar Lambert Azimuthal Equal-Area map using a continuous DTU-style colormap.

    Parameters:
    lat: 1D array of latitudes
    lon: 1D array of longitudes
    cvalue: 1D array of values
    tiepoint: value that defines the background (land) color
    colorbar_min: minimum colorbar value
    colorbar_max: maximum colorbar value
    filename: output filename
    """
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(central_latitude=90,
                                                                                    central_longitude=0)})

    ax.set_extent([-180, 180, 66.5, 90], crs=ccrs.PlateCarree())

    # Define color for land using tiepoint
    norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)
    tiepoint_color = dtu_coolwarm_cmap(norm(250)) 

    # Add land with tiepoint color
    land = cfeature.NaturalEarthFeature('physical',
                                        'land',
                                        '50m',
                                        edgecolor='face',
                                        facecolor=tiepoint_color)
    ax.add_feature(land)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}



    # Scatter plot
    sc = ax.scatter(lon, lat,
                c=cvalue,
                cmap=dtu_coolwarm_cmap,
                norm=norm,
                s=0.01,
                transform=ccrs.PlateCarree())


    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.8, pad=0.09)
    cbar.set_label('tb', fontsize=18, labelpad=15)

    plt.savefig(filename, dpi=300, bbox_inches='tight')




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
