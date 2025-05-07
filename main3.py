from footprint_operator import resample, create_img,calculate_sigmas, make_kernel
import pandas as pd
from scipy.signal import fftconvolve
import numpy as np
import imageio.v2 as imageio
#%% -------------------------- Dictonaries -------------------------- # 

# Define dictonary for CIMR IFOV (in km)
CIMR_tracks = {1.4: {"IFOV": {"along_track": 36, "cross_track": 64}},
               6.9: {"IFOV": {"along_track": 11, "cross_track": 19}},
               10.7: {"IFOV": {"along_track": 7, "cross_track": 13}},
               18.7: {"IFOV": {"along_track": 4, "cross_track": 6}},
               36.5: {"IFOV": {"along_track": 3, "cross_track": 5}}}

# Define dictonary for MWI IFOV (in km)



# Tiepoints



# -------------------------- User Input -------------------------- # 
# What is the name of the file you want to create
name='CIMR_FYI_36.5GHz_horizontal_combined_resampled'

# Set the folder path where your combined csv is and where output should be
SMRT_directory = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/CIMR_FYI_36.5GHz_horizontal_combined.csv"
output_directory =f"C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/resampled/{name}"

# OW csv with appropriate tiepoint value
OW = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/OW_161.csv"

output_norm = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/resampled/norm"
output_directory_img = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/resampled/img.png"

# Acces tracks
cross_track = CIMR_tracks[1.4]["IFOV"]["cross_track"]
along_track = CIMR_tracks[1.4]["IFOV"]["along_track"]


#%% ---------------------------------------------------------------- # 

# Merge OW and sea ice csv files
ice_df = pd.read_csv(SMRT_directory)
OW_df = pd.read_csv(OW)
df = pd.concat([OW_df, ice_df])

# Read variables
lat = df['lat']
lon = df['lon']
tb = df['tb']
print("Latitude, longitude and brightness temperatures loaded...")

# Calculate sigmas
sigma1, sigma2 = calculate_sigmas(cross_track, along_track)
print("Sigma1 and sigma2 calculated...")

# Make Gaussian kernel
kernel = make_kernel(sigma1, sigma2)
print(f"Gaussian kernel created. Dim: {kernel.shape}...")

# Resample SMRT output from 4km to 1km regular grid using nearest interpolation
grid_lon, grid_lat, grid_tb = resample(lat,lon,tb)
print("Data resampled...")

# Create image 
img_rgb, img_gray = create_img(grid_lat, grid_lon, grid_tb, 'output.png')
print(f"Image with dim: {img_rgb.shape} created and saved...")

# Apply filter. 'same' is there to enforce the same output shape as input arrays
convolved_tb = fftconvolve(img_rgb, kernel[:, :, np.newaxis], mode='same')

# Save using colormap
imageio.imwrite("convolved_tb.png", convolved_tb.astype(np.float32))
print("Gaussian kernel applied and image saved...")


