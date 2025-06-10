import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
from algorhitms import bristol_CIMR, bristol
from data_visualization import dtu_grey, dtu_red, dtu_navy,dtu_coolwarm_cmap,dtu_blues_cmap
#%%
# Load ASIP data
asip_ds = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\data\ASIP\dmi_asip_seaice_mosaic_arc_l4_20250408.nc")
lat_asip = asip_ds["lat"].values
lon_asip = asip_ds["lon"].values
sic_asip = asip_ds["sic"].isel(time=0).values
lon2d_asip, lat2d_asip = np.meshgrid(lon_asip, lat_asip) if lat_asip.ndim == 1 else (lon_asip, lat_asip)

# Load CICE
cice_ds = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\CICE\CICE_SIC.nc")
flat_lat = cice_ds["TLAT"].values.flatten()
flat_lon = cice_ds["TLON"].values.flatten()
flat_aice = cice_ds["aice"].values.flatten()
valid = ~np.isnan(flat_lat) & ~np.isnan(flat_lon) & ~np.isnan(flat_aice)
flat_lat, flat_lon, flat_aice = flat_lat[valid], flat_lon[valid], flat_aice[valid]
grid_lon = np.linspace(np.min(flat_lon), np.max(flat_lon), 500)
grid_lat = np.linspace(np.min(flat_lat), np.max(flat_lat), 500)
grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)
grid_aice = griddata((flat_lon, flat_lat), flat_aice, (grid_lon2d, grid_lat2d), method='nearest')

# Load CIMR brightness temperatures
ds_cimr_314H = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_36.5_H_CIMR.nc")
ds_cimr_314V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_36.5_V_CIMR.nc")
ds_cimr_187V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_18.7_V_CIMR.nc")
tb_cimr_37h = ds_cimr_314H["tb"].values
tb_cimr_37v = ds_cimr_314V["tb"].values
tb_cimr_18v = ds_cimr_187V["tb"].values
lat_cimr = ds_cimr_314H["lat"].values
lon_cimr = ds_cimr_314H["lon"].values
bristol_cimr = bristol_CIMR(tb_cimr_18v, tb_cimr_37v, tb_cimr_37h)
if np.nanmax(bristol_cimr) <= 1:
    bristol_cimr *= 100
lon2d_cimr, lat2d_cimr = np.meshgrid(lon_cimr, lat_cimr) if lat_cimr.ndim == 1 else (lon_cimr, lat_cimr)

# Load MWI brightness temperatures
ds_mwi_314H = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_31.4_H_MWI.nc")
ds_mwi_314V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_31.4_V_MWI.nc")
ds_mwi_187V = xr.open_dataset(r"C:\Users\user\OneDrive\Desktop\Bachelor\SMRT\Smeared\convolved_18.7_V_MWI.nc")
tb_mwi_37h = ds_mwi_314H["tb"].values
tb_mwi_37v = ds_mwi_314V["tb"].values
tb_mwi_18v = ds_mwi_187V["tb"].values
lat_mwi = ds_mwi_314H["lat"].values
lon_mwi = ds_mwi_314H["lon"].values
bristol_mwi = bristol(tb_mwi_18v, tb_mwi_37v, tb_mwi_37h)
if np.nanmax(bristol_mwi) <= 1:
    bristol_mwi *= 100
lon2d_mwi, lat2d_mwi = np.meshgrid(lon_mwi, lat_mwi) if lat_mwi.ndim == 1 else (lon_mwi, lat_mwi)

#%% Plot
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# Color intermediates
phase1_blue = '#030F4F'
phase2_blue = '#3d4677'
phase3_blue = '#babecf'
phase1_red = '#990000'
phase2_red = '#bc5959'
phase3_red = '#e6c1c1'



fig, ax = plt.subplots(figsize=(60, 5),
                       subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(central_latitude=90, central_longitude=0)})
ax.set_extent([-20, 60, 74, 78], crs=ccrs.PlateCarree())
#ax.set_extent([-180, 180, 66.5, 90], crs=ccrs.PlateCarree())

# Gridlines
# gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# gl.xlabel_style = {'size': 12}
# gl.ylabel_style = {'size': 12}
# gl.top_labels = True
# gl.bottom_labels = False
# gl.left_labels = True
# gl.right_labels = True


# Plot background (18.7V MWI AICE)
tb_background = np.ma.masked_invalid(grid_aice),
p = ax.pcolormesh(grid_lon2d, grid_lat2d, tb_background,
                  transform=ccrs.PlateCarree(),
                  cmap=dtu_blues_cmap, shading='nearest', alpha=1,
                  vmin=0, vmax=1,edgecolors='face',linewidth=0)

# Contours
# Contours with updated colors
#sic_edge = ax.contour(lon2d_asip, lat2d_asip, sic_asip, levels=[15],
#                      colors=dtu_red, linewidths=1, linestyles='--', transform=ccrs.PlateCarree())

#aice_edge = ax.contour(grid_lon2d, grid_lat2d, grid_aice, levels=[0.15],
#                       colors=phase2_red, linewidths=2, linestyles='-', transform=ccrs.PlateCarree())

cimr_edge = ax.contour(lon2d_cimr, lat2d_cimr, bristol_cimr, levels=[0.15],
                       colors=dtu_red, linewidths=1.5, linestyles='-', transform=ccrs.PlateCarree())

mwi_edge = ax.contour(lon2d_mwi, lat2d_mwi, bristol_mwi, levels=[0.15],
                      colors=dtu_red, linewidths=1.5, linestyles='--', transform=ccrs.PlateCarree())


# Add land and coastlines
land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='face', facecolor=dtu_grey)
ax.add_feature(land)
ax.coastlines(resolution='50m', linewidth=0.5, color='black')

# Add colorbar
# cb = fig.colorbar(p, ax=ax,  orientation='horizontal', pad=0.03, shrink=0.13)
# cb.set_ticks([0, 0.5, 1])
# cb.set_label('DMI-CICE SIC [Unitless]', fontsize=12, labelpad=18)
# cb.ax.tick_params(labelsize=12)

# # Legend
# legend_elements = [
#     # Line2D([0], [0], color=phase2_blue, linestyle='-', linewidth=1, label='ASIP Ice Edge'),
#     # Line2D([0], [0], color=phase3_blue, linestyle=':', linewidth=0.5, label='DMI-CICE Ice Edge'),
#     Line2D([0], [0], color=dtu_red, linestyle='-', linewidth=0.5, label='Bristol CIMR Ice Edge'),
#     Line2D([0], [0], color=dtu_red, linestyle='--', linewidth=0.5, label='Bristol MWI Ice Edge')
# ]

# ax.legend(handles=legend_elements,
#           loc='upper left',
#           frameon=True,
#           facecolor='white',
#           fontsize=12)
plt.savefig("ice_edge_comparison4.pdf", format='pdf', bbox_inches='tight')
plt.show()


