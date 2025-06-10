import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
from algorhitms import bristol_CIMR

# Load ASIP data
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
# Load datasets
ds_314H = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_36.5_H_CIMR.nc")
ds_314V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_36.5_V_CIMR.nc")
ds_187V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_18.7_V_CIMR.nc")
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

# Plotting
fig, ax = plt.subplots(figsize=(15, 5),
                       subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                           central_latitude=90, central_longitude=0)})

ax.set_extent([-30, 72, 75, 80], crs=ccrs.PlateCarree())

# Add land and coastlines
dtu_grey = "#d3d3d3"
land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='face', facecolor=dtu_grey, zorder=1)
ax.add_feature(land)
ax.coastlines(resolution='50m', linewidth=0.5, color='black')

# Gridlines
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}
gl.top_labels = True
gl.bottom_labels = False
gl.left_labels = True
gl.right_labels = True

# Plot ASIP SIC 15% contour
sic_edge = ax.contour(lon2d, lat2d, sic, levels=[15],
                      colors='black', linewidths=1.2,
                      transform=ccrs.PlateCarree())

# Plot CICE AICE 15% contour
aice_edge = ax.contour(grid_lon2d, grid_lat2d, grid_aice, levels=[0.15],
                       colors='red', linewidths=1.2,
                       transform=ccrs.PlateCarree())

# Plot CIMR Bristol SIC 15% contour
bristol_edge = ax.contour(lon_bristol, lat_bristol, bristol_SIC_footprint, levels=[0.15],
                          colors='blue', linewidths=1.2,
                          transform=ccrs.PlateCarree())

# Build legend
legend_handles = []
if sic_edge.collections:
    legend_handles.append(Line2D([0], [0], color='black', lw=1.2, label='15% SIC Edge (ASIP)'))
if aice_edge.collections:
    legend_handles.append(Line2D([0], [0], color='red', lw=1.2, label='15% AICE Edge (CICE)'))
if bristol_edge.collections:
    legend_handles.append(Line2D([0], [0], color='blue', lw=1.2, label='15% SIC Edge (CIMR Bristol)'))

ax.legend(handles=legend_handles, loc="upper left", fontsize=12)

plt.show()
