import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data_visualization import dtu_grey, dtu_blues_cmap # assuming you're not using dtu_coolwarm_cmap for plotting
import cartopy.crs as ccrs
import numpy as np
from pyproj import CRS, Transformer
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
import cv2

import numpy as np
import os
from PIL import Image
from skimage.color import rgb2gray


def compute_tb(image_path, tb_min, tb_max):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if tb_min >= tb_max:
        raise ValueError("tb_min must be smaller than tb_max.")

    rgb_img = Image.open(image_path).convert("RGB")
    rgb_array = np.array(rgb_img).astype(np.float32) / 255.0
    gray_intensity = rgb2gray(rgb_array)  # already between 0 and 1

    # Optional debug:
    print("Grayscale intensity range:", gray_intensity.min(), gray_intensity.max())

    # Direct linear scaling (no per-image normalization)
    tb_map = tb_min + gray_intensity * (tb_max - tb_min)

    return tb_map




def CICE_SIC_img(lat, lon, cvalue, output_path='plot.png'):
    """
    Plot scatter data on a North Polar Lambert Azimuthal Equal-Area map using a continuous DTU-style colormap,
    with land background and no visible axis or colorbar.

    Parameters:
    lat: 1D array of latitudes
    lon: 1D array of longitudes
    cvalue: 1D array of values
    colorbar_min: minimum colorbar value
    colorbar_max: maximum colorbar value
    output_path: output filename
    """
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(central_latitude=90,
                                                                                    central_longitude=0)})

    ax.set_extent([-180, 180, 68.5, 90], crs=ccrs.PlateCarree())

    norm = Normalize(vmin=0, vmax=1)
    tiepoint_color = dtu_blues_cmap(norm(0.292))  # Adjust if a different tiepoint is preferred

    land = cfeature.NaturalEarthFeature('physical', 
                                        'land', 
                                        '50m',
                                        edgecolor='face', 
                                        facecolor=tiepoint_color)
    ax.add_feature(land)

    ax.scatter(lon, lat,
               c=cvalue,
               cmap=dtu_blues_cmap,
               norm=norm,
               s=0.1,
               transform=ccrs.PlateCarree())

    ax.axis('off')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    # Read and convert image
    img_bgr = cv2.imread(output_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_rgb

def compute_SIC(img_rgb, SIC_min=0, SIC_max=1):
    """
    Recover SIC values from a grayscale-encoded RGB image using linear mapping.
    
    Parameters:
    img_rgb: np.ndarray of shape (H, W, 3), uint8 or float32
    SIC_min: minimum of the original data range
    SIC_max: maximum of the original data range
    
    Returns:
    sic_map: 2D np.ndarray of shape (H, W) with recovered SIC values
    """
    # Ensure float format in [0, 1]
    rgb_norm = img_rgb.astype(np.float32) / 255.0

    # Convert to grayscale intensity
    gray_intensity = rgb2gray(rgb_norm)  # Uses luminance: 0.2125 R + 0.7154 G + 0.0721 B

    # Map intensity to SIC values linearly
    sic_map = SIC_min + gray_intensity * (SIC_max - SIC_min)

    return sic_map
