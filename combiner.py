import os  
import glob  
import pandas as pd

#%% -------------------------- User Input --------------------------

# Set the folder path containing CSV files
directory_365GHz_h = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/CIMR/FYI/36.5GHz"

# Set the desired name for the combined CSV (without .csv extension here)
name_365GHz_h = 'CIMR_FYI_36.5GHz_horizontal_combined' 

# ----------------------------------------------------------------
# Define full output path including file name
output_directory = f'{directory_365GHz_h}/{name_365GHz_h}.csv' 

def combine_csv_files(directory, output_directory):
    # Find all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))  
    
    # Read and concatenate all CSVs into one DataFrame
    combined_df = pd.concat([pd.read_csv(i) for i in csv_files], ignore_index=True)  
    
    # Save the combined DataFrame to the specified output file, without row indices
    combined_df.to_csv(output_directory, index=False)  

    return combined_df  

# Call the function to combine files and save the result
csv = combine_csv_files(directory_365GHz_h, output_directory)  
