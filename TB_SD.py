

import matplotlib.pyplot as plt 
from data_visualization import dtu_coolwarm_cmap, dtu_grey, dtu_blues_cmap,dtu_reds_cmap
from dictionaries import CIMR
from algorhitms import bristol,bristol_CIMR
from skimage.transform import downscale_local_mean
from footprint_operator import resample
import pandas as pd
import xarray as xr
from dictionaries import MWI,OW_tiepoints
import numpy as np
from scipy.spatial import cKDTree

# Load SMRT CSVs
path_187V = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\MWI\MWI_18.7GHz_V_merged.csv"
path_314H = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\MWI\MWI_31.4GHz_H_merged.csv"
path_314V = r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\MWI\MWI_31.4GHz_V_merged.csv"

df_187V = pd.read_csv(path_187V)
df_314H = pd.read_csv(path_314H)
df_314V = pd.read_csv(path_314V)

lat_smrt = df_187V['lat'].values
lon_smrt = df_187V['lon'].values
tb18v = df_187V['tb'].values
tb37h = df_314H['tb'].values
tb37v = df_314V['tb'].values


# Load CICE data
df_cice = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CICEv3.csv")
df_cice = df_cice.rename(columns={'TLAT': 'lat', 'TLON': 'lon'})


# Convert lat/lon to Cartesian for nearest-neighbor interpolation
def latlon_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)

smrt_cart = latlon_to_cartesian(lat_smrt, lon_smrt)
cice_cart = latlon_to_cartesian(df_cice['lat'].values, df_cice['lon'].values)

# KDTree interpolation (16 nearest neighbors with inverse distance weighting)
tree = cKDTree(smrt_cart)
distances, indices = tree.query(cice_cart, k=16)
weights = 1 / (distances + 1e-12)
tb18v = np.average(tb18v [indices], axis=1, weights=weights)

# Build DataFrame for correlation and plotting
df_grid2 = pd.DataFrame({
    'lat': df_cice['lat'],
    'lon': df_cice['lon'],
    'tb18v': tb18v,
    'hs': df_cice['hs']
})

import matplotlib.pyplot as plt


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Reshape for sklearn
X = df_grid2['hs'].values.reshape(-1, 1)
y = df_grid2['tb18v'].values

# Linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# R² score
r2 = model.score(X, y)

# Scatterplot with regression line
plt.figure()
plt.scatter(df_grid2['hs'], df_grid2['tb18v'], s=10, label='Data')
plt.plot(df_grid2['hs'], y_pred, color='red', label=f'Fit: $R^2$ = {r2:.2f}')
plt.xlabel('hs (m)')
plt.ylabel('tb18v (K)')
plt.title('Scatterplot of hs vs. tb18v with Linear Fit')
plt.legend()
plt.grid(True)
plt.show()


#%%#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# DTU color scheme
dtu_navy = '#030F4F'
dtu_red = '#990000'
dtu_grey = '#DADADA'
white = '#ffffff'
black = '#000000'

# Updated Color intermediates
# Light to dark: lower to higher frequency
phase1_red = '#e6c1c1'  # Lightest red
phase2_red = '#bc5959'  # Medium red
phase3_red = '#990000'  # Darker red
phase4_red = '#660000'  # Darkest red

phase1_blue = '#7EA6E0'  # Lightest blue
phase2_blue = '#365F91'  # Medium blue
phase3_blue = '#002147'  # Darkest blue

# Updated color map: low freq light, high freq dark
color_map = {
    'CIMR 6.9V': phase1_red,   # 6.9 GHz - light red
    'CIMR 18.7V': phase2_red,  # 18.7 GHz - medium red
    'CIMR 36.5H': phase4_red,  # 36.5 GHz - dark red
    'CIMR 36.5V': phase3_red,  # 36.5 GHz - darker red
    'MWI 18.7V': phase1_blue,  # 18.7 GHz - light blue
    'MWI 31.4V': phase2_blue,  # 31.4 GHz - medium blue
    'MWI 31.4H': phase3_blue   # 31.4 GHz - dark blue
}

# Paths
paths = {
    'CIMR 6.9V': r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CIMR\CIMR_6.9GHz_V_merged.csv",
    'CIMR 18.7V': r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CIMR\CIMR_18.7GHz_V_merged.csv",
    'CIMR 36.5H': r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CIMR\CIMR_36.5GHz_H_merged.csv",
    'CIMR 36.5V': r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CIMR\CIMR_36.5GHz_V_merged.csv",
    'MWI 31.4H': r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\MWI\MWI_31.4GHz_H_merged.csv",
    'MWI 31.4V': r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\MWI\MWI_31.4GHz_V_merged.csv",
    'MWI 18.7V': r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\MWI\MWI_18.7GHz_V_merged.csv"
}

# Load CICE data
df_cice = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\CICEv3.csv")
df_cice = df_cice.rename(columns={'TLAT': 'lat', 'TLON': 'lon'})

# Filter CICE by iage > 0.602
df_cice = df_cice[df_cice['iage'] > 0.602]

# Convert hs from meters to centimeters and filter hs between 0 and 60 cm
df_cice['hs_cm'] = df_cice['hs'] * 100
df_cice = df_cice[(df_cice['hs_cm'] >= 0) & (df_cice['hs_cm'] <= 60)]

# Lat/lon to Cartesian
def latlon_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)

cice_cart = latlon_to_cartesian(df_cice['lat'].values, df_cice['lon'].values)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Prepare figure
fig, ax1 = plt.subplots(figsize=(10, 7))

# Set font size globally
plt.rcParams.update({'font.size': 24})

# For histogram (only once, since hs is always the same)
ax2 = ax1.twinx()
bins_hist = np.arange(0, 61, 5)  # 5 cm bin width
ax2.hist(df_cice['hs_cm'], bins=bins_hist, color=dtu_grey, edgecolor=black, alpha=0.3, label='hs histogram')

for title, file_path in paths.items():
    df = pd.read_csv(file_path)
    lat_smrt = df['lat'].values
    lon_smrt = df['lon'].values
    tb = df['tb'].values

    smrt_cart = latlon_to_cartesian(lat_smrt, lon_smrt)

    tree = cKDTree(smrt_cart)
    distances, indices = tree.query(cice_cart, k=16)
    weights = 1 / (distances + 1e-12)
    tb_interp = np.average(tb[indices], axis=1, weights=weights)

    df_grid = pd.DataFrame({
        'lat': df_cice['lat'],
        'lon': df_cice['lon'],
        'tb': tb_interp,
        'hs': df_cice['hs_cm']
    })

    bins = np.arange(0, 61, 5)  # 5 cm bin width
    df_grid['bin'] = pd.cut(df_grid['hs'], bins, labels=False)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    mean_tb_per_bin = df_grid.groupby('bin')['tb'].mean()

    valid_bins = mean_tb_per_bin.index.notna()
    bin_indices = mean_tb_per_bin.index[valid_bins].astype(int)

    x = bin_centers[bin_indices]
    y = mean_tb_per_bin.values[valid_bins]

    if 'MWI' in title:
        linestyle = '-'
    else:
        linestyle = '--'

    ax1.plot(x, y, label=title, linestyle=linestyle, color=color_map[title])

# Final plot settings
ax1.set_xlabel('DMI-CICE $h_s$ [cm]', fontsize=24)
ax1.set_ylabel('Simulated $T_B$ [K]', fontsize=24)
ax1.set_xlim(0, 60)
ax2.set_ylim(10000, 350000)

# Set ticks font size
ax1.tick_params(axis='x', which='major', labelsize=24)
#ax2.tick_params(axis='both', which='major', labelsize=24)

# Hide only y-axis tick labels of ax1
ax2.set_yticklabels([])

ax1.legend(loc='upper left', fontsize=22)

plt.show()
