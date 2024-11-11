import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_antarctica():
    # Define the projection, focused on the South Pole
    projection = ccrs.SouthPolarStereo()

    # Set up the figure and axis with the projection
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': projection})
    ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

    # Add coastlines and land features
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Optionally, add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False  # Only show bottom and left labels

    # Title
    plt.title("Antarctica", fontsize=14)

    plt.show()

# Call the function to plot Antarctica
plot_antarctica()
