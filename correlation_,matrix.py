import numpy as np
import matplotlib.pyplot as plt
from data_visualization import dtu_reds_cmap

# Updated Data
data1 = np.array([
    [1.000, 0.982, 0.959],
    [0.982, 1, 0.973],
    [0.959, 0.973, 1.000]
])

data2 = np.array([
    [1.000, 0.983, 0.901],
    [0.983, 1, 0.902],
    [0.901, 0.902, 1.000]
])

labels = ['DMI-CICE', 'SMRT', 'Smeared']

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[20, 1.2], hspace=0.1)

# Function to determine text color based on brightness
def get_text_color(value, cmap):
    r, g, b, _ = cmap(value)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if brightness < 0.6 else 'black'

# Matrix 1
ax1 = fig.add_subplot(gs[0, 0])
img1 = ax1.imshow(data1, cmap=dtu_reds_cmap, vmin=0.9, vmax=1)
ax1.set_xticks(np.arange(len(labels)))
ax1.set_yticks(np.arange(len(labels)))
ax1.set_xticklabels(labels, fontsize=30)
ax1.set_yticklabels(labels, fontsize=30)
ax1.set_title("CIMR", fontsize=30)
for i in range(len(labels)):
    for j in range(len(labels)):
        val = data1[i, j]
        ax1.text(j, i, f'{val:.3f}', ha='center', va='center',
                 color=get_text_color((val - 0.9) / 0.1, dtu_reds_cmap), fontsize=30)

# Matrix 2
ax2 = fig.add_subplot(gs[0, 1])
img2 = ax2.imshow(data2, cmap=dtu_reds_cmap, vmin=0.9, vmax=1)
ax2.set_xticks(np.arange(len(labels)))
ax2.set_xticklabels(labels, fontsize=30)
ax2.set_yticks([])
ax2.set_yticklabels([])
ax2.set_title("MWI", fontsize=30)
for i in range(len(labels)):
    for j in range(len(labels)):
        val = data2[i, j]
        ax2.text(j, i, f'{val:.3f}', ha='center', va='center',
                 color=get_text_color((val - 0.9) / 0.1, dtu_reds_cmap), fontsize=30)

# Colorbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cbar_ax = inset_axes(ax2, width="38%", height="5%", loc='lower center', bbox_to_anchor=(-0.49, 0, 2, 1),
                     bbox_transform=fig.transFigure, borderpad=0)
cbar = fig.colorbar(img1, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Correlation', fontsize=32)
cbar.ax.tick_params(labelsize=30)

plt.show()
