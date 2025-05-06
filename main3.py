from footprint_operator import normalize, upsample, create_img,calculate_sigmas, make_kernel
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from data_visualization import dtu_coolwarm_cmap
from scipy.signal import fftconvolve
from scipy import signal
import numpy as np
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

# Find max and min
tb_min = df['tb'].min()
tb_max = df['tb'].max()

# Normalize to acces tb values later
normalized_df = normalize(df,tb_min,tb_max)
print(f"Brightness temperature has been normalized using max: {tb_max} and min: {tb_min}... ")

# Calculate sigmas
sigma1, sigma2 = calculate_sigmas(cross_track, along_track)
print("Sigma1 and sigma2 calculated...")

# Make Gaussian kernel
kernel = make_kernel(sigma1, sigma2)
kernel_norm = kernel / np.sum(kernel)
print(f"Normalized Gaussian kernel created. Dim: {kernel_norm.shape}...")

# Resample SMRT output from 4km to 1km regular grid using nearest interpolation
upsampled_df = upsample(normalized_df,output_directory)
print("Data resampled...")

# Load features
upsampled_lat = upsampled_df['lat'][:10]
upsampled_lon = upsampled_df['lon'][:10]
upsampled_tb = upsampled_df['tb'][:10]
print("Variables loaded...")

#%% Create image 
img_gray, img_rbg = create_img(upsampled_lat, upsampled_lon, upsampled_tb, output_directory_img)
print(f"Image with dim: {img_gray.shape} created and saved...")

#%% Apply filter. 'same' is there to enforce the same output shape as input arrays
convolved_tb = fftconvolve(img_gray, kernel_norm, mode='same')

# Save using colormap
plt.imsave('convolved_tb.png', convolved_tb)
print("Gaussian kernel applied and image saved...")

#%%
print(img_gray.shape)

