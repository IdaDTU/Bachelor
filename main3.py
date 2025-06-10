from footprint_operator import resample, create_img,calculate_sigmas, make_kernel, save_tb_to_netcdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from scipy.signal import convolve2d
from scipy import signal
from scipy.signal import fftconvolve
from data_visualization import dtu_coolwarm_cmap
import imageio
from dictionaries import CIMR_tracks, MWI_tracks

# -------------------------- User Input -------------------------- # 
# The paths to the combined simulation for a given freq, sensor and ice type 
freq, pol = 36.5, "H"
name =f"{freq}_{pol}_CIMR"
SMRT_directory = f"/zhome/da/d/187040/bachelor/csv/footprint/36.5_H_CIMR"

# OW csv with appropriate tiepoint value for 0W
OW = f"/zhome/da/d/187040/bachelor/csv/footprint/OW_145.29"

# Acces tracks
cross_track = CIMR_tracks[freq]["IFOV"]["cross_track"]
along_track = CIMR_tracks[freq]["IFOV"]["along_track"]

#%% ---------------------------------------------------------------- # 
print(freq)
print(pol)

# Merge OW and sea ice csv files
ice_df = pd.read_csv(SMRT_directory)
OW_df = pd.read_csv(OW)

# Merge with sea ice priority (overwrites OW if same lat/lon exists)
df = pd.concat([OW_df, ice_df]).drop_duplicates(subset=['lat', 'lon'], keep='last')

# Read variables
lat = df['lat']
lon = df['lon']
tb = df['tb']
print("Latitude, longitude and brightness temperatures loaded...")

# Resample SMRT output from 4km to 1km regular grid using nearest interpolation
grid_lon, grid_lat, grid_tb = resample(lat, lon, tb)
print("Data resampled...")

# Access max and minimum brightness temperatures
tb_min = np.nanmin(grid_tb)
tb_max = np.nanmax(grid_tb)
print(f"Maximum: {tb_max} and Minimum: {tb_min} computed...")

# Calculate sigmas
sigma1, sigma2 = cross_track, along_track
print("Sigma1 and sigma2 calculated...")

# Make Gaussian kernel
kernel = make_kernel(sigma1, sigma2)
print(f"Gaussian kernel created. Dim: {kernel.shape}...")

# Apply filter directly to physical brightness temperature data
convolved_tb = fftconvolve(grid_tb, kernel, mode='same')
print("Kernel applied to brightness temperature...")

# Save filtered brightness temperature as image for visualization
# Use min/max from original tb for consistent colormap scaling
img_rgb, img_gray = create_img(grid_lat, grid_lon, convolved_tb, 'convolved_tb.png')
print("Convolved brightness temperature image saved...")

# (Optional) Save filtered temperature field as NetCDF
save_tb_to_netcdf(convolved_tb, grid_lat, grid_lon, "convolved_tb.nc")
print("Filtered brightness temperature saved to NetCDF.")
