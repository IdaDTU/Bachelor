import os  
import glob  
import pandas as pd

#%% -------------------------- User Input --------------------------

# Set the folder path containing CSV files from SMRT
SMRT_directory = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/SMRT/combine"

# Set folder for CICE and OW
OW_input_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/OW"
CICE_directory = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/"
tiepoint_OW = 145

# Set the desired name for the combined CSV (without .csv extension here)
name_SMRT= 'CIMR_FYI_36.5GHz_horizontal_combined' 


# ----------------------------------------------------------------
# Define full output path including file name
SMRT_output_directory = f'{SMRT_directory}/{name_SMRT}.csv' 
CICE_output_directory = f'{CICE_directory}/OW_{tiepoint_OW}.csv'

def combine_csv_files(directory, output_directory):
    # Find all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))  
    
    # Read and concatenate all CSVs into one DataFrame
    combined_df = pd.concat([pd.read_csv(i) for i in csv_files], ignore_index=True)  
    
    # Save the combined DataFrame to the specified output file, without row indices
    combined_df.to_csv(SMRT_output_directory, index=False)  

    return combined_df  

def add_OW_column(OW_input_path, tiepoint, CICE_output_directory):
    # Load the data
    df = pd.read_csv(OW_input_path)

    # Add the tb column
    df["tb"] = tiepoint

    # Keep only TLAT, TLON, and tb
    df = df[["tb","TLAT", "TLON"]]

    # Rename TLAT and TLON
    df = df.rename(columns={"TLAT": "lat", "TLON": "lon"})

    # Save the result
    df.to_csv(CICE_output_directory, index=False)
    return df


    
# Combine SMRT
#csv = combine_csv_files(SMRT_directory, SMRT_output_directory)  

# Make OW columns in CICE csv
add_OW_column(OW_input_path, tiepoint_OW, CICE_output_directory)
