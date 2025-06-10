import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from data_visualization import dtu_grey, dtu_red

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

def temperature_gradient_snow(thickness_snow, thickness_ice, temperature_air, layers_snow):
    ksi = 2.1
    ks = 0.3
    temperature_water = 271.15

    interface_temp = (ksi * thickness_snow * temperature_water + ks * thickness_ice * temperature_air) / \
                     (ksi * thickness_snow + ks * thickness_ice)
    
    gradient = (interface_temp - temperature_air) / thickness_snow

    snow_depths = np.linspace(0, thickness_snow, layers_snow)
    snow_temp = temperature_air + gradient * snow_depths
    snow_temp = np.minimum(snow_temp, 273.15)

    ice_depths = np.linspace(thickness_snow, thickness_snow + thickness_ice, 3)
    ice_temp = np.linspace(snow_temp[-1], temperature_water, 3)

    full_depth = np.concatenate([snow_depths, ice_depths[1:]])
    full_temp = np.concatenate([snow_temp, ice_temp[1:]])

    return full_temp, full_depth, thickness_snow, thickness_ice

# Load CSV
CICE = r"C:\Users\user\OneDrive\Desktop\Bachelor\CSV\CICE\CICE"
df = pd.read_csv(CICE)

# Mean values
thickness_snow = df['hs'].mean()
thickness_ice = df['hi'].mean()
temperature_air = df['tair'].mean()

# Profile
layers_snow = 3
temps, depths, h_snow, h_ice = temperature_gradient_snow(thickness_snow, thickness_ice, temperature_air, layers_snow)

fig, (ax_main, ax_layer) = plt.subplots(
    ncols=2,
    figsize=(8, 6),
    gridspec_kw={'width_ratios': [5, 0.6], 'wspace': 0.05},
    sharey=True
)

# --- Temperature-depth profile ---
ax_main.axhspan(0, h_snow, facecolor='white', alpha=1, label='Snow')  # now blue
ax_main.axhspan(h_snow, h_snow + h_ice, facecolor='#d3e6f5', alpha=1, label='Ice')  # now light blue
ax_main.axhline(h_snow, color='black', linestyle='--', linewidth=1, label='Snow-Ice Interface')
ax_main.plot(temps, depths, color=dtu_red, linewidth=2, label='Temperature Profile')

ax_main.set_xlabel("Temperature [K]")
ax_main.set_ylabel("Depth [m]")
ax_main.set_xlim(254, 273.15)
ax_main.set_ylim(0, h_snow + h_ice)
ax_main.invert_yaxis()
ax_main.legend(loc='lower left')

# --- Layer structure plot ---
n_snow_layers = 3
snow_edges = np.linspace(0, h_snow, n_snow_layers + 1)
for i in range(n_snow_layers):
    ax_layer.fill_betweenx(
        [snow_edges[i], snow_edges[i + 1]],
        0.25, 0.75,
        color='white',  # now blue
        edgecolor='black'
    )

n_ice_layers = 2
ice_edges = np.linspace(h_snow, h_snow + h_ice, n_ice_layers + 1)
for i in range(n_ice_layers):
    ax_layer.fill_betweenx(
        [ice_edges[i], ice_edges[i + 1]],
        0.25, 0.75,
        color='#d3e6f5',  # now light blue
        edgecolor='black'
    )

ax_layer.set_xlim(0, 1)
ax_layer.set_xticks([])
ax_layer.set_ylabel("")
ax_layer.set_xlabel("Layers")
ax_layer.set_ylim(0, h_snow + h_ice)

ax_layer.tick_params(left=False, labelleft=False)
ax_layer.spines['left'].set_visible(False)
ax_layer.spines[['top', 'right', 'bottom']].set_visible(False)

ax_layer.invert_yaxis()

plt.tight_layout()
plt.show()

