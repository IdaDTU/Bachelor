import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#%%


def footprint_4km_to_1km_grid(csv_path):
    """
    Interpolates scattered CSV data to a 1x1 km regular lat/lon grid and saves the result to a new CSV file.

    Parameters:
        csv_path (str): Path to CSV file containing 'CICE_TLAT', 'CICE_TLON', 'SMRT_tbh'.
        basemap_plot (bool): Whether to plot the data on a geographic map using Cartopy.

    Returns:
        DataFrame with columns: lat, lon, interpolated tb.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract columns
    lat = df["CICE_TLAT"].values
    lon = df["CICE_TLON"].values
    var = df["SMRT_tbh"].values

    # Prepare interpolation input
    points = np.column_stack((lon, lat))

    # Define resolution
    res_km = 1
    res_lat = res_km / 111.32
    lat_center = np.mean(lat)
    res_lon = res_km / (111.32 * np.cos(np.radians(lat_center)))

    # Define target grid
    lon_new = np.arange(np.min(lon), np.max(lon), res_lon)
    lat_new = np.arange(np.min(lat), np.max(lat), res_lat)
    lon_grid, lat_grid = np.meshgrid(lon_new, lat_new)

    # Interpolate
    var_interp = griddata(points, var, (lon_grid, lat_grid), method='linear')

    # Flatten and create output DataFrame
    df_out = pd.DataFrame({
        "lat": lat_grid.flatten(),
        "lon": lon_grid.flatten(),
        "tb": var_interp.flatten()})

    # Remove NaNs if needed
    df_out = df_out.dropna()

    # Save output
    df_out.to_csv('C:/Users/user/OneDrive/Desktop/Bachelor/csv/1km_CICE.csv', index=False)

    return df_out



# %%sigma
# def calculate_sigmas(cross_track,along_track):
#     sigma_along = along_track/2.355
#     sigma_cross = cross_track/2.355
#     return sigma_along,sigma_cross


# print(calculate_sigmas(3,5))
    
# #%%

# csv_path = "C:/Users/user/OneDrive/Desktop/Bachelor/CSV/1km.csv"
# df = pd.read_csv(csv_path)
# print(df)

# import pandas as pd
# from scipy.signal import convolve2d

# def make_kernel(sigma1, sigma2):
#     # t is just x axis?
#     t = np.linspace(-300, 300, 600)
    
#     # Make two normal distribution for sigma1 and sigma2, meaning 2 1D kernels
#     pdf1 = (1.0 / np.sqrt(2 * np.pi * sigma1**2)) * np.exp(-t**2 / (2 * sigma1**2))
#     pdf2 = (1.0 / np.sqrt(2 * np.pi * sigma2**2)) * np.exp(-t**2 / (2 * sigma2**2))
    
#     # Calculate area underneath curve, using trapezoidal rule and normalise
#     pdf1  /= np.trapz(pdf1)
#     pdf2 /= np.trapz(pdf2)
    
#     # Create 2D kernel from 1D kernel
#     kernel = pdf1[:, np.newaxis] * pdf2[np.newaxis, :]
#     return kernel

# #%%
# from scipy import signal
# kernel37v_cimr=make_kernel(calculate_sigmas(3,5))

# #Ivanova et al. southern hemisphere OW FY tie-points for AMSRE
# tp={'fy6v':257.04,  'my6v': 254.18, 'ow6v':159.69,  'fy6h': 236.52, 'my6v': 225.37, 'ow6h': 80.15,\
#     'fy10v':257.23, 'my10v':251.65, 'ow10v':166.31, 'fy10h':238.50, 'my10h':221.47, 'ow10h':86.62,\
#     'fy18v':258.58, 'my18v':246.10, 'ow18v':185.34, 'fy18h':242.80, 'my18h':217.65, 'ow18h':110.83,\
#     'fy22v':257.56, 'my22v':240.65, 'ow22v':201.53, 'fy22h':242.61, 'my22h':213.79, 'ow22h':137.19,\
#     'fy37v':253.84, 'my37v':226.51, 'ow37v':212.57, 'fy37h':239.96, 'my37h':204.66, 'ow37h':149.07,\
#     'fy90v':242.81, 'my90v':210.22, 'ow90v':247.59, 'fy90h':232.40, 'my90h':197.78, 'ow90h':207.20}

# 1km_res = plt.imread('Antarctica.A2008055.0330.250m.png')

# #values inbetween 0 and 1: brightness to SIC
# lower=(1km_res < 0.01)
# upper=(1km_res > 0.99)
# 1km_res[lower]=0.0
# 1km_res[upper]=1.0    
    
# tb37v = (1.0-1km_res) * tp['ow37v'] + 1km_res * tp['fy37v']
# cimr37v = signal.fftconvolve(tb37v, kernel37v_cimr[:, :, np.newaxis], mode='same')

# #%%

# # Load CSV as 2D array
# data = df.to_numpy()

# # Create Gaussian kernel
# kernel = make_kernel(sigma1=2.1231422505307855, sigma2=1.2738853503184713)

# # Convolve data with kernel (same size output)
# smoothed = convolve2d(data, kernel, mode='same', boundary='symm')

# # Optional: Save smoothed data
# smoothed_df = pd.DataFrame(smoothed).to_csv("smoothed_output.csv", index=False, header=False)
# print(smoothed_df)

# #%% Optional: visualize
# plt.imshow(smoothed, cmap='viridis')
# plt.title("Smoothed Data")
# plt.colorbar()
# plt.show()





