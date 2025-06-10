# Import necessary modules
import os
import glob
import pandas as pd
from dictionaries import OW_tiepoints
# Set the folder path containing CSV files from SMRT
directory = r'C:\Users\user\OneDrive\Desktop\Bachelor\CSV\SMRT\combine'

# Set folder for CICE and OW
CICE_path = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CICEv3.csv"
# Read CICE data
cice = pd.read_csv(CICE_path)

# Define full output path including file name
# Set the desired name for the combined CSV (without .csv extension here)
name_SMRT = 'CIMR_MYI_36.5GHz_H_combined'
output_directory = f'{directory}/{name_SMRT}.csv'


def combine_csv_files(directory, output_directory, cice):
    csv_files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    combined_df = pd.concat([pd.read_csv(i) for i in csv_files], ignore_index=True)
    combined_sorted = combined_df.sort_values(['lat', 'lon']).reset_index(drop=True)
    cice = cice.reset_index(drop=True)
    combined_sorted = combined_sorted[cice['aice'] > 0.15].reset_index(drop=True)
    cice = cice[cice['aice'] > 0.15].reset_index(drop=True)
    tp = OW_tiepoints['36.5H']
    combined_sorted['tb_scaled'] = (1 - cice['aice']) * tp + cice['aice'] * combined_sorted['tb']
    combined_sorted.drop(columns=['tb'], inplace=True)
    combined_sorted.rename(columns={'tb_scaled': 'tb'}, inplace=True)
    combined_sorted.to_csv(output_directory, index=False)
    return combined_df

def add_OW_column(OW_input_path, tiepoint, CICE_output_directory):
    df = pd.read_csv(OW_input_path)
    df["tb"] = tiepoint
    df = df[["tb", "TLAT", "TLON"]]
    df = df.rename(columns={"TLAT": "lat", "TLON": "lon"})
    df.to_csv(CICE_output_directory, index=False)
    return df

def add_OW_column2(OW_input_path, sic, CICE_output_directory):
    df = pd.read_csv(OW_input_path)
    df["aice"] = sic
    df = df[["aice", "TLAT", "TLON"]]
    df = df.rename(columns={"TLAT": "lat", "TLON": "lon"})
    df.to_csv(CICE_output_directory, index=False)
    return df

def merge_iage(FYI_path, MYI_path, CICE_path, output_csv_path):
    CICE = pd.read_csv(CICE_path)
    FYI = pd.read_csv(FYI_path)
    MYI = pd.read_csv(MYI_path)
    CICE = CICE.rename(columns={'TLAT': 'lat', 'TLON': 'lon'})
    CICE = CICE[['lat', 'lon', 'iage']]
    FYI_merged = FYI.merge(CICE, on=['lat', 'lon'], how='left')
    MYI_merged = MYI.merge(CICE, on=['lat', 'lon'], how='left')
    FYI_filtered = FYI_merged[FYI_merged['iage'] <= 0.602]
    MYI_filtered = MYI_merged[MYI_merged['iage'] > 0.602]
    result = pd.concat([FYI_filtered, MYI_filtered], ignore_index=True)
    result = result[['lat', 'lon', 'tb', 'iage']]
    result.to_csv(output_csv_path, index=False)
    return result

#%%
combine_csv_files(directory, output_directory, cice)
#%%

# Example call (assuming you have a `cice` DataFrame loaded)
FYI_path = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\SMRT\combine\18.7_H_FY_CIMR_combined_new.csv"
MYI_path = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\SMRT\combine\18.7_H_MY_CIMR_combined_new.csv"
CICE_path = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CICEv3.csv"
output_csv_path =  r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\SMRT\combine\CIMR_18.7H.csv"
merge_iage(FYI_path, MYI_path, CICE_path, output_csv_path)

#%%
# Set folder for CICE and OW
CICE_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/CICE/CICEv3.csv"
OW_input_path = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\CICE\OWv3"
tiepoint = OW_tiepoints['31.4V']
CICE_output_directory = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\OW_31.4V.csv"

def add_OW_column(OW_input_path, tiepoint, CICE_output_directory):
    df = pd.read_csv(OW_input_path)
    df["tb"] = tiepoint
    df = df[["tb", "TLAT", "TLON"]]
    df = df.rename(columns={"TLAT": "lat", "TLON": "lon"})
    df.to_csv(CICE_output_directory, index=False)
    return df

#%%
sic = CICE_path

add_OW_column2(OW_input_path, sic, CICE_output_directory)




#%%
OW_input_path = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\OWv3.2"
sic = 0
CICE_output_directory = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\OW_0SIC.csv"
def add_OW_column2(OW_input_path, sic, CICE_output_directory):
    df = pd.read_csv(OW_input_path)
    df["aice"] = sic
    df = df[["aice", "TLAT", "TLON"]]
    df = df.rename(columns={"TLAT": "lat", "TLON": "lon"})
    df.to_csv(CICE_output_directory, index=False)
    return df

add_OW_column2(OW_input_path, tiepoint, CICE_output_directory)





