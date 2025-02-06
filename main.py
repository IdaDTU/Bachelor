from init_sensor_snowpack import init_sensor_snowpack

# Dataframes
MWI_df = init_sensor_snowpack(sensor='MWI',
                     thickness=[100],
                     corr_length=[5e-5],
                     microstructure_model='exponential',
                     density=320)


CIMR_df = init_sensor_snowpack(sensor='CIMR',
                     thickness=[100],
                     corr_length=[5e-5],
                     microstructure_model='exponential',
                     density=320)

#print(CIMR_df)
#print(MWI_df)