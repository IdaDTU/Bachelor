import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
from algorhitms import bristol_CIMR
from data_visualization import dtu_red,dtu_blues_cmap,dtu_grey

#%% Load ASIP data
ASIP_l3 = r"C:\Users\user\OneDrive\Desktop\Bachelor\data\ASIP\dmi_asip_seaice_mosaic_arc_l4_20250408.nc"
asip_ds = xr.open_dataset(ASIP_l3)

lat = asip_ds["lat"].values
lon = asip_ds["lon"].values
sic = asip_ds["sic"].isel(time=0).values

if lat.ndim == 1 and lon.ndim == 1:
    lon2d, lat2d = np.meshgrid(lon, lat)
else:
    lon2d, lat2d = lon, lat

# Load CICE netCDF properly
CICE_nc = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\CICE\CICE_SIC.nc"
cice_ds = xr.open_dataset(CICE_nc)

cice_lat = cice_ds["TLAT"].values
cice_lon = cice_ds["TLON"].values
cice_aice = cice_ds["aice"].values

# Load CIMR Bristol
ds_314H = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_36.5_H_CIMR.nc")
ds_314V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_36.5_V_CIMR.nc")
ds_187V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\convolved_18.7_V_CIMR.nc")
print("Footprints loaded...")

# Extract tb arrays
tb37h = ds_314H["tb"].values
tb37v = ds_314V["tb"].values
tb18v = ds_187V["tb"].values
print("Brightness temperatures extracted...")

lat_cimr = ds_314H["lat"].values
lon_cimr = ds_314H["lon"].values
print("Lat and lon extracted...")

# Compute SIC with Bristol
bristol_SIC_footprint = bristol_CIMR(tb18v, tb37v, tb37h)
print("Bristol SIC computed...")


# Prepare lat/lon grid for Bristol data
if lat_cimr.ndim == 1 and lon_cimr.ndim == 1:
    lon_bristol, lat_bristol = np.meshgrid(lon_cimr, lat_cimr)
else:
    lon_bristol, lat_bristol = lon_cimr, lat_cimr

# Flatten for interpolation of CICE
flat_lat = cice_lat.flatten()
flat_lon = cice_lon.flatten()
flat_aice = cice_aice.flatten()

# Remove NaNs before interpolation
valid = ~np.isnan(flat_lat) & ~np.isnan(flat_lon) & ~np.isnan(flat_aice)
flat_lat = flat_lat[valid]
flat_lon = flat_lon[valid]
flat_aice = flat_aice[valid]

# Interpolate CICE aice to grid
grid_lon = np.linspace(np.min(flat_lon), np.max(flat_lon), 500)
grid_lat = np.linspace(np.min(flat_lat), np.max(flat_lat), 500)
grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

grid_aice = griddata(
    (flat_lon, flat_lat), flat_aice,
    (grid_lon2d, grid_lat2d),
    method='nearest'
)

#%% Plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import numpy as np

fig, ax = plt.subplots(figsize=(10, 10),
                       subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                           central_latitude=90, central_longitude=0)})

ax.set_extent([-180, 180, 68, 90], crs=ccrs.PlateCarree())

# Plot aice as filled contour background
cice_plot = ax.pcolormesh(grid_lon2d, grid_lat2d, grid_aice, 
                          transform=ccrs.PlateCarree(), 
                          cmap=dtu_blues_cmap,
                          zorder=1)

# Add colorbar
cbar = fig.colorbar(cice_plot, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8)
cbar.set_ticks([0, 0.5, 1])
cbar.set_label('DMI-CICE SIC [Unitless]', fontsize=22, labelpad=10)
cbar.ax.tick_params(labelsize=20)

sic_edge = ax.contour(lon2d, lat2d, sic, levels=[15],
                      colors=dtu_red, linewidths=2, linestyles='--',
                      transform=ccrs.PlateCarree(), zorder=2)

# Plot CICE AICE 15% contour
aice_edge = ax.contour(grid_lon2d, grid_lat2d, grid_aice, levels=[0.15],
                       colors=dtu_red, linewidths=2,
                       transform=ccrs.PlateCarree(), zorder=2)

# Add land
land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='face', facecolor=dtu_grey)
ax.add_feature(land, zorder=3)

# Add coastlines
ax.coastlines(resolution='50m', linewidth=0.5, color='black', zorder=3)

# --- MANUAL POLAR GRID ---

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=0.8, color='gray', alpha=0, linestyle='--')
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
gl.top_labels = True
gl.bottom_labels = False
gl.left_labels = False
gl.right_labels = True

# Define parallels (circles)
parallels = np.arange(40, 91, 5)  # 60°N to 90°N
theta = np.linspace(0, 360, 60)  # 0° to 360° for full circle
for lat in parallels:
    r = 90 - lat  # distance from pole
    ax.plot(theta, [lat]*len(theta), transform=ccrs.PlateCarree(),
            color='gray', linewidth=0.8, linestyle='-', alpha=0.7, zorder=5)

# Define meridians (lines from pole)
meridians = np.arange(-180, 180, 60)  # every 30°
radii = np.linspace(50, 90, 100)
for lon in meridians:
    ax.plot([lon]*len(radii), radii, transform=ccrs.PlateCarree(),
            color='gray', linewidth=0.8, linestyle='-', alpha=0.7, zorder=5)

# Build legend
legend_handles = []
if sic_edge.collections:
    legend_handles.append(Line2D([0], [0], color=dtu_red, lw=1.2, label='ASIP Ice Edge'))
if aice_edge.collections:
    legend_handles.append(Line2D([0], [0], color=dtu_red,linestyle='--', lw=1.2, label='DMI-CICE Ice Edge'))

ax.legend(handles=legend_handles, loc="upper left", fontsize=20)

plt.show()
