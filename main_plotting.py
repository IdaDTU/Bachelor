# Load modules
from data_visualization import plot_npstere_categorical, plot_npstere_cmap
import numpy as np
import pandas as pd

#%% -------------------------- User Input -------------------------- # 

# CICE paths
cice_input_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE"
cice_output_path = "C:/Users/user/OneDrive/Desktop/Bachelor/plots/CICE_NCAT.png"

SMRT_input_path ="C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/36.5GHz/Horizontal/combined/CIMR_FYI_36.5GHz_horizontal_combined.csv"
input_path_365GHz_h ="C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/36.5GHz/CIMR_FYI_36.5GHz_horizontal_combined.csv"
#%% ---------------------------------------------------------------- # 

# CICE plots
CICE_df = pd.read_csv(cice_input_path)

# Extract CICE data for plots
CICE_lat = CICE_df['TLAT']
CICE_lon = CICE_df['TLON']
CICE_hi = CICE_df['hi']
CICE_aicen = CICE_df['aicen'] > 0.15

# Call plotting functions
#plot_npstere_categorical(CICE_lat, CICE_lon, CICE_hi ,cice_output_path)
plot_npstere_cmap(CICE_lat, CICE_lon, CICE_aicen, colorbar_min=0, colorbar_max=1, filename='plot.pdf')
#%%


#%%  
# SMRT plots

SMRT_df = pd.read_csv(input_path_365GHz_h)
SMRT_lat = SMRT_df['lat']
SMRT_lon = SMRT_df['lon']
SMRT_TB = SMRT_df['tb']
colorbar_min = 180
colorbar_max = 240


plot_npstere_cmap(SMRT_lat, SMRT_lon, SMRT_TB, colorbar_min, colorbar_max, filename='plot.pdf')

#%%
SMRT_lat = SMRT_df['lat']
SMRT_lon = SMRT_df['lon']

print(SMRT_lat.max())