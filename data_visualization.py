import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def plot_measurements(lat,
                      lon, 
                      cvalue):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Set up the map with North Polar Stereographic projection
    # lon_0,lat_0 is central point
    m = Basemap(projection='npstere',
                boundinglat=66.5,   # Only show latitudes north of 66.5° N
                lon_0=0,   # Central meridian; adjust as needed
                resolution='l',
                ax=ax,
                round = True)
    
    # Draw a thicker circular map boundary.
    m.drawmapboundary(linewidth=2, color='black')

    # Draw map features
    m.drawcoastlines()
    m.drawparallels(np.arange(50, 90, 10), labels=[True, True, False, False])
    m.drawmeridians(np.arange(-180, 180, 30), labels=[True, False, False, True])
    
    # Fill land and ocean
    m.fillcontinents(color='white', lake_color='lightblue')
    
    # Convert lat/lon to map coordinates
    x, y = m(lon, lat)

    # Scatter plot
    sc = ax.scatter(x, y, c=cvalue, cmap='coolwarm', s=100, alpha=0.8)
    plt.colorbar(sc, ax=ax, orientation='vertical', label='Measurement Value')
    
    # Set title
    ax.set_title("Arctic Circle", fontsize=14)
    
    # Save before showing
    plt.savefig('plot.pdf')
   

