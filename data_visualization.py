import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from scipy.stats import linregress

def plot_measurements(lat, lon, colorbar_min, colorbar_max, colorbar_label, title, color, cvalue=None):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Set up the map with North Polar Stereographic projection
    # lon_0,lat_0 is central point
    m = Basemap(projection='npstere',
                boundinglat=70,   # Only show latitudes north of 66.5° N
                lon_0=0,   # Central meridian; adjust as needed
                resolution='l',
                ax=ax,
                round = True)

    # Draw map features
    m.drawcoastlines()
    m.drawparallels(np.arange(50, 90, 10), labels=[True, True, False, False])
    m.drawmeridians(np.arange(-180, 180, 30), labels=[True, False, False, True])
    
    OW_tiepoint = 161.35
    
    # Create lat/lon grid based on map boundary (boundinglat=70)
    OW_lon = np.linspace(-180, 180, 360)  # Longitude from -180 to 180
    OW_lat = np.linspace(70, 90, 180)  # Latitude from 70 to 90

    # Create meshgrid for lat/lon
    OW_lon_grid, OW_lat_grid = np.meshgrid(OW_lon, OW_lat)

    # Convert the lat/lon grid to map projection coordinates (x, y)
    OW_x, OW_y = m(OW_lon_grid, OW_lat_grid)

    # Convert lat/lon to map coordinates
    x, y = m(lon, lat)
    
    OW_plot = np.full_like(OW_x, OW_tiepoint, dtype=float)
    
    
    sc = ax.scatter(OW_x, OW_y, c=OW_plot, cmap=color, s=10, alpha=0.8, vmin = colorbar_min, vmax = colorbar_max)
    
    # Scatter plot
    if cvalue is not None:
        ax.scatter(x, y, c=cvalue, cmap=color, s=10, alpha=0.8, vmin = colorbar_min, vmax = colorbar_max)
        plt.colorbar(sc, ax=ax, orientation='vertical', label=colorbar_label)
    else:
        ax.scatter(x, y, color='red', edgecolors='k', s=10, alpha=0.8, vmin = colorbar_min, vmax = colorbar_max)  # Default single color

    # Fill land
    m.fillcontinents(color='darkgrey', lake_color='lightblue')    

    # Set title
    ax.set_title(title, fontsize=14)

    # Show the plot
    #plt.savefig("/zhome/57/6/168999/Desktop/plot.png", dpi=300, bbox_inches="tight") #jose hpc path
    #plt.savefig("/zhome/da/d/187040/plot.pdf", dpi=300, bbox_inches="tight") #ida hpc path
    plt.show() # used for Spyder

def plot_sensitivity(n, variable, tbv, title, xlabel, ylabel, labels, name):
    plt.figure()
    
    for i in range(len(tbv)):
        tbv_i = tbv[i].values.ravel()
        print(f"variable[:n]: shape={variable[:n].shape}")
        print(f"tbv[i]: shape={tbv[i].shape}")
        #plt.scatter(variable[:n], tbv[i], label=labels[i])
        slope, intercept, r_value, p_value, std_err = linregress(variable[:n], tbv_i)
        plt.plot(variable[:n], slope * variable[:n] + intercept, label='f{labels[i]}: y={slope:.2f}x + {intercept:.2f}')
        
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    
    plt.savefig("/zhome/57/6/168999/Desktop/plots/sensitivity_plot_{name}.png", dpi=300, bbox_inches="tight")
    #plt.show()
    

