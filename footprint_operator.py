import numpy as np
from scipy.interpolate import griddata
import matplotlib as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
from data_visualization import dtu_coolwarm_cmap,dtu_grey
from pyproj import CRS, Transformer
from matplotlib.colors import LinearSegmentedColormap, Normalize,ListedColormap, BoundaryNorm
import xarray as xr

def make_kernel(sigma1, sigma2):
    # t is just x axis?
    t = np.linspace(-50, 50, 100)
    
    # Make two normal distribution for sigma1 and sigma2, meaning 2 1D kernels
    pdf1 = (1.0 / np.sqrt(2 * np.pi * sigma1**2)) * np.exp(-t**2 / (2 * sigma1**2))
    pdf2 = (1.0 / np.sqrt(2 * np.pi * sigma2**2)) * np.exp(-t**2 / (2 * sigma2**2))
    
    # Calculate area underneath curve, using trapezoidal rule and normalise
    pdf1  /= np.trapz(pdf1)
    pdf2 /= np.trapz(pdf2)
    
    # Create 2D kernel from 1D kernel
    kernel = pdf1[:, np.newaxis] * pdf2[np.newaxis, :]
    return kernel

def resample(lat, lon, tb, center_lat=90, center_lon=0):
    """
    Resamples scattered lat/lon data to a 1x1 km grid in LAEA projection.

    Parameters:
        df (DataFrame): Input data with lat, lon, and value columns
        lat_col (str): Name of the latitude column
        lon_col (str): Name of the longitude column
        value_col (str): Name of the variable to interpolate (e.g., brightness temperature)
        center_lat (float): Central latitude of LAEA projection
        center_lon (float): Central longitude of LAEA projection

    Returns:
        Tuple of (grid_lon, grid_lat, grid_value)
    """
    # Define projection
    laea_crs = CRS.from_proj4(f"+proj=laea +lat_0={center_lat} +lon_0={center_lon} +x_0=0 +y_0=0 +units=m +ellps=WGS84 +no_defs")
    wgs84_crs = CRS.from_epsg(4326)

    transformer = Transformer.from_crs(wgs84_crs, laea_crs, always_xy=True)
    x, y = transformer.transform(lon.values, lat.values)

    # Define 1 km grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    grid_x, grid_y = np.meshgrid( np.arange(x_min, x_max + 1000, 1000),
                                 np.arange(y_min, y_max + 1000, 1000))

    # Interpolate
    grid_value = griddata((x, y),
                          tb.values,
                          (grid_x, grid_y),
                          method='linear')

    # Convert back to lat/lon
    inverse_transformer = Transformer.from_crs(laea_crs, wgs84_crs, always_xy=True)
    grid_lon, grid_lat = inverse_transformer.transform(grid_x, grid_y)

    return grid_lon, grid_lat, grid_value

def create_img(grid_lat, grid_lon, grid_tb, output_path):
    """
    Plots gridded data using Lambert Azimuthal Equal-Area projection centered on North Pole.

    Parameters:
        grid_lat (ndarray): 2D array of latitudes
        grid_lon (ndarray): 2D array of longitudes
        grid_tb (ndarray): 2D array of data values
        output_path (str): File path to save the image
        dtu_coolwarm_cmap: Colormap to use (assumed to be defined globally)
        dtu_grey: Land facecolor (assumed to be defined globally)
    """
    vmin = np.nanmin(grid_tb)
    vmax = np.nanmax(grid_tb)
    
    projection = ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=90)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300, subplot_kw={'projection': projection})

    ax.set_extent([-180, 180, 66.5, 90], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(grid_lon, grid_lat, grid_tb,
                         cmap=dtu_coolwarm_cmap,
                         shading='auto',
                         vmin=vmin,
                         vmax=vmax,
                         transform=ccrs.PlateCarree())
    
    # Define color for land using tiepoint
    norm = Normalize(vmin, vmax)
    tiepoint_color = dtu_coolwarm_cmap(norm(250)) 
    land_feature = cfeature.NaturalEarthFeature(category='physical',
                                                name='land',
                                                scale='10m',
                                                facecolor=tiepoint_color)
    ax.add_feature(land_feature, zorder=10)

    ax.axis('off')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img_bgr = cv2.imread(output_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return img_rgb, img_gray

def save_tb_to_netcdf(grid_tb, grid_lat, grid_lon, output_path):
	# Save to NetCDF
    ds = xr.Dataset(data_vars={"tb": (("y", "x"), grid_tb)},
        coords={"lat": (("y", "x"), grid_lat),
				"lon": (("y", "x"), grid_lon)},
        attrs={"title": "Synthetic Brightness Temperature Map",
				"units": "Kelvin",
				"projection": "Lambert Azimuthal Equal Area"})
    ds.to_netcdf(output_path)

