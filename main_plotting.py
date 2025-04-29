# Load modules
from data_visualization import plot_npstere_categorical, plot_npstere_cmap
import numpy as np
import pandas as pd

#%% -------------------------- User Input -------------------------- # 

# CICE paths
cice_input_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE"
cice_output_path = "C:/Users/user/OneDrive/Desktop/Bachelor/plots/CICE_NCAT.pdf"

SMRT_input_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/1.4GHz/Vertical/combined/CIMR_FYI_1.4GHz_vertical_combined.csv"
#%% ---------------------------------------------------------------- # 

# CICE plots
CICE_df = pd.read_csv(cice_input_path)

# Extract CICE data for plots
CICE_lat = CICE_df['ULAT']
CICE_lat = CICE_df['ULON']
CICE_CATS = CICE_df['NCAT']

# Call plotting function
plot_npstere_categorical(CICE_lat, CICE_lat, CICE_CATS ,cice_output_path)


#%%  
# SMRT plots

SMRT_df = pd.read_csv(SMRT_input_path)
SMRT_lat = SMRT_df['lat']
SMRT_lon = SMRT_df['lon']
SMRT_TB = SMRT_df['tb']
tiepoint = 159
colorbar_min = 200
colorbar_max = 300


plot_npstere_cmap(SMRT_lat, SMRT_lon, SMRT_TB, tiepoint, colorbar_min, colorbar_max, filename='plot.pdf')

#%%
SMRT_lat = SMRT_df['lat']
SMRT_lon = SMRT_df['lon']

print(SMRT_lon.max())