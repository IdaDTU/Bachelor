import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.preprocessing import MinMaxScaler
from data_visualization import dtu_coolwarm_cmap,dtu_grey
import cv2


def normalize(df,tb_min,tb_max):
    scaler = MinMaxScaler()
    df['tb'] = scaler.fit_transform(df[['tb']])
    return df

def upsample(df, output_path):
    """
    Interpolates scattered CSV data to a 1x1 km regular lat/lon grid and saves the result to a new CSV file.

    Parameters:
        csv_path (str): Path to CSV file containing 'lat', 'lon', 'tb' columns.
        name (str): Filename to save interpolated CSV as.
        output_dir (str): Directory to save the output CSV file.
        method (str): Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        DataFrame with columns: lat, lon, interpolated tb.
    """

    # Extract columns
    lat = df["lat"].values
    lon = df["lon"].values
    var = df["tb"].values

    # Prepare interpolation input
    points = np.column_stack((lon, lat))

    # Define resolution
    res_km = 1
    km_per_deg = 111.195
    res_lat = res_km / km_per_deg
    lat_center = np.mean(lat)
    res_lon = res_km / (km_per_deg * np.cos(np.radians(lat_center)))

    # Define target grid
    lon_new = np.arange(np.min(lon), np.max(lon) + res_lon, res_lon)
    lat_new = np.arange(np.min(lat), np.max(lat) + res_lat, res_lat)
    lon_grid, lat_grid = np.meshgrid(lon_new, lat_new)

    # Interpolate
    var_interp = griddata(points, var, (lon_grid, lat_grid), method='linear')

    # Flatten and create output DataFrame
    df_out = pd.DataFrame({"lat": lat_grid.ravel(),
                           "lon": lon_grid.ravel(),
                           "tb": var_interp.ravel()})

    # Remove NaNs
    df_out = df_out.dropna()
    
    # Save output
    df_out.to_csv(output_path, index=False)

    return df_out

def calculate_sigmas(cross_track,along_track):
     sigma_along = along_track/2.355
     sigma_cross = cross_track/2.355
     return sigma_along,sigma_cross

def make_kernel(sigma1, sigma2):
    # t is just x axis?
    t = np.linspace(-300, 300, 600)
    
    # Make two normal distribution for sigma1 and sigma2, meaning 2 1D kernels
    pdf1 = (1.0 / np.sqrt(2 * np.pi * sigma1**2)) * np.exp(-t**2 / (2 * sigma1**2))
    pdf2 = (1.0 / np.sqrt(2 * np.pi * sigma2**2)) * np.exp(-t**2 / (2 * sigma2**2))
    
    # Calculate area underneath curve, using trapezoidal rule and normalise
    pdf1  /= np.trapz(pdf1)
    pdf2 /= np.trapz(pdf2)
    
    # Create 2D kernel from 1D kernel
    kernel = pdf1[:, np.newaxis] * pdf2[np.newaxis, :]
    return kernel

def create_img(lat, lon, tb, output_path):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300, subplot_kw={'projection': ccrs.NorthPolarStereo()})
    ax.set_extent([-180, 180, 66.5, 90], crs=ccrs.PlateCarree())

    # Use your desired colormap (e.g., dtu_coolwarm_cmap)
    sc = ax.scatter(lon, lat, c=tb, cmap=dtu_coolwarm_cmap, s=0.01, transform=ccrs.PlateCarree())

    ax.coastlines()
    land_feature = cfeature.NaturalEarthFeature(category='physical',
                                                name='land',
                                                scale='10m',
                                                facecolor=dtu_grey)
    ax.add_feature(land_feature, zorder=10)

    ax.axis('off')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Read in both RGB and grayscale versions
    img_bgr = cv2.imread(output_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return img_rgb, img_gray


