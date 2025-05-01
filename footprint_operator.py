import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import os
import matplotlib as plt
import cartopy.crs as ccrs
from io import BytesIO
from PIL import Image

def footprint_4km_to_1km_grid(csv_path, name, output_dir, method='linear'):
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
    # Load CSV
    df = pd.read_csv(csv_path)

    # Check required columns
    for col in ['lat', 'lon', 'tb']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

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
    var_interp = griddata(points, var, (lon_grid, lat_grid), method=method)

    # Flatten and create output DataFrame
    df_out = pd.DataFrame({"lat": lat_grid.ravel(),
                           "lon": lon_grid.ravel(),
                           "tb": var_interp.ravel()})

    # Remove NaNs
    df_out = df_out.dropna()

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name)

    # Save output
    df_out.to_csv(output_path, index=False)

    return df_out

def resampled_to_image(lat, lon, cvalue, filename):
    """
    Plot scatter data on a North Polar Stereographic map using a grayscale colormap (Cartopy version),
    save it to a file, and return the image as a 2D NumPy array (grayscale, 0-255 uint8).

    Parameters:
    lat: 1D array of latitudes
    lon: 1D array of longitudes
    cvalue: 1D array of values
    filename: output filename (including extension, e.g., .png or .jpg)

    Returns:
    img_array: 2D NumPy array of the saved image (grayscale)
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.NorthPolarStereo()})
    ax.set_extent([-180, 180, 65.5, 90], crs=ccrs.PlateCarree())

    ax.scatter(lon, lat, c=cvalue, cmap='gray', s=0.01, transform=ccrs.PlateCarree())

    plt.savefig(filename, dpi=300, bbox_inches='tight')

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    img = Image.open(buf)
    img_array = np.array(img)

    plt.close(fig)
    buf.close()

    # Convert to grayscale if necessary
    if img_array.ndim == 3:
        img_array = img_array[..., :3]  # Drop alpha channel if it exists
        img_array = np.dot(img_array, [0.2989, 0.5870, 0.1140])  # Convert RGB to grayscale

    if img_array.max() <= 1.0:
        img_array = img_array * 255

    img_array = img_array.astype(np.uint8)

    return img_array

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


#%%














#%%


# # Load an example image
# image = plt.imread('your_image.png')  # make sure to replace this with your image path

# # If it's a color image, convert to grayscale
# if image.ndim == 3:
#     image = np.mean(image, axis=2)

# # Create kernel
# kernel = make_kernel(sigma1=10, sigma2=20)

# # Convolve image with kernel
# blurred_image = scipy.signal.convolve2d(image, kernel, mode='same', boundary='symm')

# # Plot original and blurred image
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Blurred Image')
# plt.imshow(blurred_image, cmap='gray')
# plt.axis('off')

# plt.show()





# #%%
# from scipy import signal
# kernel37v_cimr=make_kernel(calculate_sigmas(3,5))

# #Ivanova et al. southern hemisphere OW FY tie-points for AMSRE
# tp={'fy6v':257.04,  'my6v': 254.18, 'ow6v':159.69,  'fy6h': 236.52, 'my6v': 225.37, 'ow6h': 80.15,\
#     'fy10v':257.23, 'my10v':251.65, 'ow10v':166.31, 'fy10h':238.50, 'my10h':221.47, 'ow10h':86.62,\
#     'fy18v':258.58, 'my18v':246.10, 'ow18v':185.34, 'fy18h':242.80, 'my18h':217.65, 'ow18h':110.83,\
#     'fy22v':257.56, 'my22v':240.65, 'ow22v':201.53, 'fy22h':242.61, 'my22h':213.79, 'ow22h':137.19,\
#     'fy37v':253.84, 'my37v':226.51, 'ow37v':212.57, 'fy37h':239.96, 'my37h':204.66, 'ow37h':149.07,\
#     'fy90v':242.81, 'my90v':210.22, 'ow90v':247.59, 'fy90h':232.40, 'my90h':197.78, 'ow90h':207.20}

# 1km_res = plt.imread('Antarctica.A2008055.0330.250m.png')

# #values inbetween 0 and 1: brightness to SIC
# lower=(1km_res < 0.01)
# upper=(1km_res > 0.99)
# 1km_res[lower]=0.0
# 1km_res[upper]=1.0    
    
# tb37v = (1.0-1km_res) * tp['ow37v'] + 1km_res * tp['fy37v']
# cimr37v = signal.fftconvolve(tb37v, kernel37v_cimr[:, :, np.newaxis], mode='same')

# #%%

# # Load CSV as 2D array
# data = df.to_numpy()

# # Create Gaussian kernel
# kernel = make_kernel(sigma1=2.1231422505307855, sigma2=1.2738853503184713)

# # Convolve data with kernel (same size output)
# smoothed = convolve2d(data, kernel, mode='same', boundary='symm')

# # Optional: Save smoothed data
# smoothed_df = pd.DataFrame(smoothed).to_csv("smoothed_output.csv", index=False, header=False)
# print(smoothed_df)

# #%% Optional: visualize
# plt.imshow(smoothed, cmap='viridis')
# plt.title("Smoothed Data")
# plt.colorbar()
# plt.show()





