
# Imports
from init_sensor_make_model import init_sensor_snowpack, init_sensor_icecolumn
from create_input_df import combine_nc_files,create_input_dataframe

# Insert .nc data directory
directory = 'C:/Users/user/OneDrive/Desktop/Bachelor/Arctic_data' 

# Combine all .nc files in directory into one
combined_nc=combine_nc_files(directory) 

# Convert to df
input_df=create_input_dataframe(combined_nc) 


#%%

print(input_df)


#%% Plot orbit

#%% SMRT outputs
MWI_output_df_snowpack = init_sensor_snowpack(sensor='MWI',
                                              thickness=[100],
                                              corr_length=[5e-5],
                                              microstructure_model='exponential',
                                              density=320)


CIMR_output_df_snowpack = init_sensor_snowpack(sensor='CIMR',
                                               thickness=[100],
                                               corr_length=[5e-5],
                                               microstructure_model='exponential',
                                               density=320)

#print(CIMR_df)
#print(MWI_df)




#%%
# local import
from smrt import make_ice_column, make_snowpack, make_model, sensor_list
from smrt import PSU
import numpy as np
#from smrt.inputs.make_medium import make_ice_column

# prepare inputs
l = 9 #9 ice layers
thickness = np.array([1.5/l] * l) #ice is 1.5m thick
corr_length=np.array([1.0e-3] * (l))
temperature = np.linspace(273.15-20., 273.15 - 1.8, l) #temperature gradient in the ice from -20 deg C at top to freezing temperature of water at bottom (-1.8 deg C)
salinity = np.linspace(2., 10., l)*PSU #salinity profile ranging from salinity=2 at the top to salinity=10 at the bottom of the ice

# create a multi-year sea ice column with assumption of spherical brine inclusions (brine_inclusion_shape="spheres"), and 10% porosity:
ice_type = 'multiyear' # first-year or multi-year sea ice
porosity = 0.08 # ice porosity in fractions, [0..1]

test = init_sensor_icecolumn(sensor='CIMR',
                            ice_type=ice_type,
                            thickness=thickness,
                            temperature=temperature,
                            microstructure_model="exponential",
                            brine_inclusion_shape="spheres", #brine_inclusion_shape can be "spheres", "random_needles" or "mix_spheres_needles"
                            salinity=salinity, #either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice 
                            porosity = porosity, # either density or 'porosity' should be set for sea ice. If porosity is given, density is calculated in the model. If none is given, ice is treated as having a porosity of 0% (no air inclusions)
                            corr_length=corr_length,
                            add_water_substrate="ocean" #see comment below
                            )

print(test)