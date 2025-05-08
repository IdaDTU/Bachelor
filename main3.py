from footprint_operator import resample, create_img,calculate_sigmas, make_kernel
import pandas as pd
from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from dictionaries import CIMR_tracks

# -------------------------- User Input -------------------------- # 
# What is the name of the file you want to create
name='CIMR_FYI_36.5GHz_horizontal_combined_resampled'

# The paths to the combined simulation for a given freq, sensor and ice type 
SMRT_directory = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/CIMR_FYI_36.5GHz_horizontal_combined.csv"
SMRT_output_directory =f"C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/resampled/{name}"

# OW csv with appropriate tiepoint value for 0W
OW = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/OW_161.csv"

# Acces tracks
cross_track = CIMR_tracks[1.4]["IFOV"]["cross_track"]
along_track = CIMR_tracks[1.4]["IFOV"]["along_track"]


#%% ---------------------------------------------------------------- # 

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
grid_lon, grid_lat, grid_tb = resample(lat,lon,tb)
print("Data resampled...")

# Acces max and minimum brightness temperatures
tb_min = np.nanmin(grid_tb)
tb_max = np.nanmax(grid_tb)
print(f"Maximum: {tb_max} and Minimum: {tb_min} computed...")

# Create image 
img_rgb, img_gray = create_img(grid_lat, grid_lon, grid_tb, 'output.png')
print(f"Image with dim: {img_rgb.shape} created and saved...")

# Calculate sigmas
sigma1, sigma2 = calculate_sigmas(cross_track, along_track)
print("Sigma1 and sigma2 calculated...")

# Make Gaussian kernel
kernel = make_kernel(sigma1, sigma2)
print(f"Gaussian kernel created. Dim: {kernel.shape}...")

# Apply filter. 'same' is there to enforce the same output shape as input arrays
convolved_tb = fftconvolve(img_rgb, kernel[:, :, np.newaxis], mode='same')
print("Kernel applied to image...")

# Normalize convolved_tb using grid_tb range
normalized = (convolved_tb - tb_min) / (tb_max - tb_min)
normalized = np.clip(normalized, 0, 1)  # ensure it's in [0, 1]
print("Brightness temperature normalized...")

# Save as RGB image
plt.imsave('convolved_tb.png', normalized)
print("Convolved image saved...")