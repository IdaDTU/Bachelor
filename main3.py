from footprint_operator import footprint_4km_to_1km_grid, resampled_to_image, calculate_sigmas,make_kernel
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import numpy as np

#%% -------------------------- User Input -------------------------- # 

# Set the folder path where your combined csv is and where output should be
input_directory = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/1.4GHz/Vertical/combined/CIMR_FYI_1.4GHz_vertical_combined.csv"
output_directory ='C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/1.4GHz/Vertical/combined/'

# What is the name of the file you want to create
name='CIMR_FYI_1.4GHz_vertical_combined_resampled'

cross_track = 3  # km or pixels
along_track = 5


# ---------------------------------------------------------------- # 

# Resample SMRT output from 4km to 1km grid using linear interpolation
# resampled_1km = footprint_4km_to_1km_grid(input_directory, 
#                                           name,
#                                           output_directory,
#                                           method='linear')

# Prepare input
# lat = resampled_1km

#%% Create image 
#resampled_to_image(lat, lon, cvalue, filename)

# Calculate sigmas
sigma1, sigma2 = calculate_sigmas(cross_track, along_track)

# Make Gaussian kernel
kernel = make_kernel(sigma1, sigma2)

#%% Load resampled data 
tb = plt.imread('C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/1.4GHz/Vertical/combined/CIMR_FYI_1.4GHz_vertical_combined_resampled_img.png')

#%% Apply convolution
# 'same' is there to enforce the same output shape as input arrays
tb_smoothed = convolve2d(tb, kernel, mode='same', boundary='symm')

# Plot results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(tb, cmap='gray')
plt.colorbar(label='Tb [K]')

plt.subplot(1, 2, 2)
plt.title('Smoothed')
plt.imshow(tb_smoothed, cmap='gray')
plt.colorbar(label='Tb [K]')
plt.tight_layout()
plt.show()
