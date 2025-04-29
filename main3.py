from data_preparation import combine_csv_files
from footprint_operator import footprint_4km_to_1km_grid

#%% -------------------------- User Input -------------------------- # 

# Set the folder path where your CSVs are
directory = 'C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/MY/18.7GHz'

# Set name of combined csv
combined_name = 'CIMR_MY_18.7GHz'

# ---------------------------------------------------------------- # 

# Define output-directory, dependent on directory
output_directory = f'{directory}/combined/{combined_name}_combined.csv'

# Combine csv files
csv = combine_csv_files(directory)



# Resample CICE from 4km to 1km grid