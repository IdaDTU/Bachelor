import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def plot_measurements(lat, lon, cvalue=None):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Set up the map with North Polar Stereographic projection
    # lon_0,lat_0 is central point
    m = Basemap(projection='npstere',
                boundinglat=66.5,   # Only show latitudes north of 66.5° N
                lon_0=0,   # Central meridian; adjust as needed
                resolution='l',
                ax=ax)


    # Draw map features
    m.drawcoastlines()
    m.drawparallels(np.arange(50, 90, 10), labels=[True, True, False, False])
    m.drawmeridians(np.arange(-180, 180, 30), labels=[True, False, False, True])
    
    # Fill land and ocean
    #m.fillcontinents(color='lightgray', lake_color='lightblue')
    #m.drawmapboundary(fill_color='lightblue'
    m.bluemarble()
    
    # Convert lat/lon to map coordinates
    x, y = m(lon, lat)

    # Scatter plot
    if cvalue is not None:
        sc = ax.scatter(x, y, c=cvalue, cmap='viridis', s=10, alpha=0.8)
        plt.colorbar(sc, ax=ax, orientation='vertical', label='Measurement Value')
    else:
        ax.scatter(x, y, color='red', edgecolors='k', s=10, alpha=0.8)  # Default single color

    # Set title
    ax.set_title("Arctic Region Measurements", fontsize=14)

    # Show the plot
    plt.show()

