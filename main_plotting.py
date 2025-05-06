# Load modules
from data_visualization import plot_npstere_categorical, plot_npstere_cmap, dtu_navy, plot_regular
import numpy as np
import pandas as pd

#%% -------------------------- User Input -------------------------- # 

# CICE paths
cice_input_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/CICE"
cice_output_path = "C:/Users/user/OneDrive/Desktop/Bachelor/plots/CICE_NCAT.png"
cice_unfiltered_input_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/CICE_unfiltered"
# SMRT paths
input_path_365GHz_h ="C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/CIMR_FYI_36.5GHz_horizontal_combined.csv"
OW_161 = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/OW_161.csv"
OW_145 = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/OW_145.csv"

#%% Resampled data paths
resampled_df = pd.read_csv("C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/resampled/CIMR_FYI_36.5GHz_horizontal_combined_resampled")

#%% ---------------------------------------------------------------- # 

# CICE plots
CICE_df = pd.read_csv(cice_input_path)

# Extract CICE data for plots
CICE_lat = CICE_df['TLAT']
CICE_lon = CICE_df['TLON']
CICE_hi = CICE_df['hi']

colorbar_min = 0
colorbar_max = 3

# Call plotting functions
plot_npstere_categorical(CICE_lat, CICE_lon, CICE_hi ,cice_output_path)
#plot_npstere_cmap(CICE_lat, CICE_lon, CICE_hi, colorbar_min, colorbar_max, filename='plot.pdf')

#%%
CICE_hi = CICE_df['iage']
print(CICE_hi.min())

#%%  
# SMRT plots
# Load and merge two CSVs
ice = pd.read_csv(input_path_365GHz_h)
OW = pd.read_csv(OW_145)

df = pd.concat([OW, ice], ignore_index=False)

SMRT_lat = df['lat']
SMRT_lon = df['lon']
SMRT_tb = df['tb']
colorbar_min = 145
colorbar_max = 210
#%%
print(len(CICE_df))
print(len(ice))
#%%
plot_npstere_cmap(df['lat'], df['lon'], df['tb'], colorbar_min, colorbar_max, filename='avg_plot.pdf')

#%%
print(CICE_df)
#%% Resampled plots
# Prepare input
resampled_lat = resampled_df['lat']
resampled_lon = resampled_df['lon']
resampled_tb = resampled_df['tb']
colorbar_min = 145
colorbar_max = 210

# plot
plot_npstere_cmap(resampled_lat, resampled_lon, resampled_tb, colorbar_min, colorbar_max, filename='plot.pdf')
