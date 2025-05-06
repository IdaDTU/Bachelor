# Imports
from data_preparation import combine_nc_files, create_combined_dataframe, remove_nonphysical

# -------------------------- User Input -------------------------- # 

# Create ice layers and snow layers
layers_ice = 2 # select amount of ice layers
layers_snow = 3 # select amount of snow layers

# Choose time step
t = 0

# Insert data directory
data_directory = 'C:/Users/user/OneDrive/Desktop/Bachelor/data/test' # Ida directory 

# Insert output directory for CICE csv
output_directory_CICE = 'C:/Users/user/OneDrive/Desktop/Bachelor/csv/CICE'
output_directory_OW = 'C:/Users/user/OneDrive/Desktop/Bachelor/csv/OW'
# ---------------------------------------------------------------- # 

# Combine all .nc files in directory into one
combined_nc=combine_nc_files(data_directory)  
print('combined_nc created...')

# Create subset of dataset
ds_subset = combined_nc.isel(time=t)
print('subset created...')

# Create dataframes
input_df =  create_combined_dataframe(ds_subset, layers_ice)
print('input_df created...')

# Remove unphysical data from dataframe
filtered_df,dropped_df = remove_nonphysical(input_df, layers_ice)
print(f'outliers removed from input_df: {len(input_df)-len(filtered_df)} ...')

# Convert to CSV
CICE_csv = filtered_df.to_csv(output_directory_CICE, index=False) 
OW_csv = dropped_df.to_csv(output_directory_OW, index=False) 
print('CICE.csv and OW.csv created...')

