from smrt import make_snowpack, make_ice_column, make_model, sensor_list, make_water_body
from create_input_df import combine_nc_files,create_input_dataframe
from smrt.substrate.reflector import make_reflector
#from create_sensor import create_sensor
from create_snowpack import create_snowpack
from create_icecolumn import create_icecolumn
from compute_smrt_tb import compute_smrt_tb 
import xarray as xr
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)

file_path = "/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Bachelorprojekt/DMI_data/iceh_inst.2024-04-01-03600.nc"
df = xr.open_dataset(file_path)
input_df = create_input_dataframe(df)

#%%

icelayer = create_icecolumn(ice_type = 'multiyear',
                      number_of_layers = 5,
                      thickness_ice = input_df['hi'],
                      thickness_snow = input_df['hs'],
                      temperature_air = input_df['tair'],
                      temperature_water = input_df['sst'],
                      temperature_ice = input_df['tinz']
                      )

print(icelayer)

#%%

snowpack = create_snowpack(number_of_layers = 2,
                    thickness_ice=input_df['hi'],
                    thickness_snow=input_df['hs'],
                    temperature_air=input_df['tair'],
                    temperature_water=input_df['sst'])

print(snowpack)

#%% Tried to create an ocean substrate but python refuses
import pandas as pd
from smrt.inputs.make_medium import make_water_layer, make_medium

# ✅ Create a valid water layer
water_layer = make_water_layer(
    layer_thickness=10,  
    temperature=271.15,  
    salinity=32
)

# ✅ Create a Pandas DataFrame with ALL required columns
water_data = pd.DataFrame({
    'thickness': [10],  # Required for SMRT
    'temperature': [271.35],
    'salinity': [32],
    'microstructure_model': ['homogeneous'],  # ✅ NEW: Required by SMRT!
    'density': [1025]  # Optional but useful for water
})

# ✅ Wrap it inside make_medium() with correct data
water_medium = make_medium(
    layers=[water_layer], 
    data=water_data  # ✅ Now contains 'microstructure_model'
)

print(f"✅ Type of water_medium: {type(water_medium)}")  # Should NOT be `Layer`

#%% reflector substrate works

# Ensure icelayer has no substrate before adding
icelayer.substrate = None 
snowpack.substrate = None 

medium = snowpack + icelayer

substrate = make_reflector(temperature=265, specular_reflection=0.02)

# Assign the proper medium as substrate
medium.substrate = substrate  # ✅ Now SMRT should recognize it

#%% computing the brightness temeprature
TB = compute_smrt_tb('MWI','89',medium)

print(TB)







