
# Imports
#from init_sensor_make_model import init_sensor_snowpack, init_sensor_icecolumn
from data_preparation import combine_nc_files,create_input_dataframe, remove_nonphysical, temperature_gradient_snow
from measurement_visualizations import plot_measurements, plot_sensitivity
import pandas as pd
from SMRT import SMRT_create_ice_columns, SMRT_create_snowpacks, SMRT_create_mediums, SMRT_create_sensor
from smrt import make_ice_column, make_snowpack, make_model, sensor_list

# Insert .nc data directory
#directory = 'C:/Users/user/OneDrive/Desktop/Bachelor/test' # Ida directory 
#directory = '/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Bachelorprojekt/DMI_data' # Josephine directory
directory = '/zhome/57/6/168999/Desktop/DMI_data'

# Combine all .nc files in directory into one
combined_nc=combine_nc_files(directory)  
print('combined_nc created...')

# Create subset of dataset
ds_subset = combined_nc.isel(time = 0)  # select time step
print('subset created...')

layers_ice = 2
layers_snow = 3

# Create dataframe
input_df = create_input_dataframe(ds_subset, layers_ice)
print('input_df created...')

#%% Remove outliers dataframe
filtered_df = remove_nonphysical(input_df, layers_ice)
print('amount removed from input_df:', len(input_df)-len(filtered_df))

# Extract and define relevant variabels 
thickness_ice = filtered_df['hi'] # in m
thickness_snow = filtered_df['hs'] # in m
temperature_air = filtered_df['tair']
salinity = filtered_df['sice']
ice_age = filtered_df['iage']
time = filtered_df['time']
print('variables extracted...')

temperature_profile_ice = filtered_df['temperature_profiles'] # in K
salinity_profile_ice = filtered_df['salinity_profiles'] # in kg/kg

temperature_profile_snow = temperature_gradient_snow(thickness_snow,
							thickness_ice,
							temperature_air,
							layers_snow)
							
n = len(filtered_df)
step = 100

ice_columns = SMRT_create_ice_columns(thickness_ice,
                                      temperature_profile_ice,
                                      salinity_profile_ice,
                                      n,
                                      layers_ice)
                                      
print('ice columns created')

snowpacks = SMRT_create_snowpacks(thickness_snow,
                                  temperature_profile_snow,
                                  n,
                                  layers_snow)
                                  
print('snowpacks created')

mediums = SMRT_create_mediums(snowpacks,
                              ice_columns,
                              n)

print('mediums created')

# Initialize results list
results1 = []
results2 = []
results3 = []
results4 = []
results5 = []

for i in range(0, len(mediums),step):
    
    # Create the sensor object
    sensor1 = sensor_list.passive(1.4e9, 55.0)
    sensor2 = sensor_list.passive(6.9e9, 55.0)
    sensor3 = sensor_list.passive(10.65e9, 55.0)
    sensor4 = sensor_list.passive(18.7e9, 55.0)
    sensor5 = sensor_list.passive(36.5e9, 55.0)
    
    # Set solver options
    n_max_stream = 128
    m = make_model("iba", "dort", rtsolver_options={"n_max_stream": n_max_stream, "error_handling":"nan"})
    
    # Run the model
    res1 = m.run(sensor1, mediums[i])
    res2 = m.run(sensor2, mediums[i])
    res3 = m.run(sensor3, mediums[i])
    res4 = m.run(sensor4, mediums[i])
    res5 = m.run(sensor5, mediums[i])
    
    # Compute TB in vertical polarization
    tbv1 = res1.TbV()
    tbv2 = res2.TbV()
    tbv3 = res3.TbV()
    tbv4 = res4.TbV()
    tbv5 = res5.TbV()
    results1.append(tbv1)
    results2.append(tbv2)
    results3.append(tbv3)
    results4.append(tbv4)
    results5.append(tbv5)

# Convert results to DataFrame
tbV1 = pd.DataFrame({'tbv1': results1})
tbV2 = pd.DataFrame({'tbv2': results2})
tbV3 = pd.DataFrame({'tbv3': results3})
tbV4 = pd.DataFrame({'tbv4': results4})
tbV5 = pd.DataFrame({'tbv5': results5})
print(f"Processed {len(tbV1)} entries.")

tbv = [tbV1, tbV2, tbV3, tbV4, tbV5]
labels = ['1.4 GHz', '6.9 GHz', '10.65 GHz', '18.7 GHz', '36.5 GHz']

plot_sensitivity(n=n, 
                 variable=thickness_ice[::step],
                 tbv=tbv,
                 title='Ice Thickness Sensitivity Study',
                 xlabel='Ice Thickness [m]',
                 ylabel='Brightness Temperature [K]',
                 labels=labels,
                 name='hi_3snow')

plot_sensitivity(n=n,
		 variable=thickness_snow[::step],
		 tbv=tbv,
		 title='Snow Thickness Sensitivity Study',
		 xlabel='Snow Thickness [m]',
		 ylabel='Brightness Temperature [K]',
		 labels=labels,
		 name='hs_3snow')
		 
plot_sensitivity(n=n,
		 variable=temperature_air[::step],
		 tbv=tbv,
		 title='Air Temperature Sensitivity Study',
		 xlabel='Air Temperature [K]',
		 ylabel='Brightness Temperature [K]',
		 labels=labels,
		 name='tair_3snow')
		 
plot_sensitivity(n=n,
		 variable=salinity[::step],
		 tbv=tbv,
		 title='Salinity Sensitivity Study',
		 xlabel='Salinity [ppt]',
		 ylabel='Brightness Temperature [K]',
		 labels=labels,
		 name='sal_3snow')




