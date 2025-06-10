import matplotlib.pyplot as plt 
from data_visualization import dtu_coolwarm_cmap, dtu_grey, dtu_blues_cmap
from dictionaries import CIMR
from algorhitms import bristol
from skimage.transform import downscale_local_mean
from footprint_operator import resample
import pandas as pd
import xarray as xr
from dictionaries import MWI,OW_tiepoints

# Load datasets
ds_314H = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_31.4_H_MWI.nc")
ds_314V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_31.4_V_MWI.nc")
ds_187V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_18.7_V_MWI.nc")
print("Footprints loaded...")

# Extract tb arrays
tb37h = ds_314H["tb"].values
tb37v = ds_314V["tb"].values
tb18v = ds_187V["tb"].values
print("Brightness temperatures extracted...")

lat = ds_314H["lat"].values
lon = ds_314H["lon"].values
print("Lat and lon extracted...")



#%% Compute SIC with Bristol
bristol_SIC_footprint = bristol(tb18v, tb37v, tb37h)
print("Bristol SIC computed...")
print(bristol_SIC_footprint.shape) # Change grid from (6811, 6277) to (1491, 1115)

#%%
# Acces CICE
CICE = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\CICE\CICE_SIC.nc")

# Get SIC
CICE_aice =CICE['aice']
CICE_lat =CICE['TLAT']
CICE_lon = CICE['TLON']
print(CICE_aice)

plt.imshow(bristol_SIC_footprint, cmap="viridis")
plt.colorbar(label="Sea Ice Concentration (fraction)")
plt.title("CICE aice field")
plt.show()

print(CICE_aice.shape)
#%%
import numpy as np
import xarray as xr

# Wrap Bristol SIC with dummy index coords
bristol_da = xr.DataArray(
    bristol_SIC_footprint,
    dims=["y", "x"],
    coords={
        "y": np.arange(bristol_SIC_footprint.shape[0]),
        "x": np.arange(bristol_SIC_footprint.shape[1])
    },
    name="bristol_SIC"
)

# Interpolate to CICE shape (index-based)
target_y = np.linspace(0, bristol_SIC_footprint.shape[0] - 1, CICE_aice.shape[0])
target_x = np.linspace(0, bristol_SIC_footprint.shape[1] - 1, CICE_aice.shape[1])

bristol_resampled = bristol_da.interp(
    y=xr.DataArray(target_y, dims="nj"),
    x=xr.DataArray(target_x, dims="ni"),
    method="linear"
)

# Clip to valid range
bristol_resampled = bristol_resampled.clip(0.0, 1.0)

# Assign CICE grid coordinates to resampled Bristol SIC
bristol_on_cice_grid = xr.DataArray(
    data=bristol_resampled.data,  # use raw values
    dims=CICE_aice.dims,
    coords={
        "nj": CICE_aice["nj"],
        "ni": CICE_aice["ni"],
        "TLAT": (("nj", "ni"), CICE_lat.data),
        "TLON": (("nj", "ni"), CICE_lon.data)
    },
    name="bristol_SIC"
)

# ✅ Done: now bristol_on_cice_grid has the CICE grid but Bristol SIC values
print(bristol_on_cice_grid)


#%%

plt.imshow(bristol_SIC_footprint, cmap="viridis")
plt.colorbar(label="Sea Ice Concentration (fraction)")
plt.title("br")
plt.show()

plt.imshow(bristol_on_cice_grid, cmap="viridis")
plt.colorbar(label="Sea Ice Concentration (fraction)")
plt.title("br")
plt.show()

plt.imshow(CICE_aice, cmap="viridis")
plt.colorbar(label="Sea Ice Concentration (fraction)")
plt.title("CICE")
plt.show()

#%%
diff = bristol_on_cice_grid - CICE_aice
plt.imshow(diff, cmap="viridis")
plt.colorbar(label="Sea Ice Concentration (fraction)")
plt.title("CICE aice field")
plt.show()

#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt


# Projection and figure
projection = ccrs.LambertAzimuthalEqualArea(central_longitude=0, central_latitude=90)
fig, ax = plt.subplots(figsize=(10, 10), dpi=300, subplot_kw={'projection': projection})

# Set Arctic extent
ax.set_extent([-180, 180, 67.5, 90], crs=ccrs.PlateCarree())

# Set inputs
grid_lon = CICE_lon
grid_lat = CICE_lat
grid_tb = diff
vmin = -1
vmax = 1

# Plot with colormap
norm = Normalize(vmin=vmin, vmax=vmax)
mesh = ax.pcolormesh(grid_lon, grid_lat, grid_tb,
                     cmap=dtu_coolwarm_cmap,
                     shading='auto',
                     vmin=vmin,
                     vmax=vmax,
                     transform=ccrs.PlateCarree())

# Land feature
land_feature = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='10m',
    edgecolor='face',
    facecolor=dtu_grey
)
ax.add_feature(land_feature, zorder=2)

# Gridlines
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
gl.top_labels = False
gl.right_labels = False

# Colorbar
cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.8, pad=0.05)
cbar.set_label('SIC Difference (Bristol - CICE)', fontsize=14)

# Turn off axis and save
ax.axis('off')
fig.savefig("C:/Users/user/OneDrive/Desktop/Bachelor/Figures/sic_diff.png",
            dpi=300, bbox_inches='tight', pad_inches=0)

plt.show()




