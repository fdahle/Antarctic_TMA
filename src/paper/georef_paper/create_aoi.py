from shapely.geometry import Polygon
import geopandas as gpd

def create_square(top_left, bottom_right):
    """
    Create a square polygon given the top left and bottom right coordinates.
    """
    return Polygon([
        top_left,
        (bottom_right[0], top_left[1]),
        bottom_right,
        (top_left[0], bottom_right[1]),
        top_left
    ])

def save_shapefile(polygon, filename):
    """
    Save the polygon to a shapefile.
    """
    epsg_code = 3031
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=f"EPSG:{epsg_code}")
    gdf.to_file(filename, driver='ESRI Shapefile')

# Define the top left and bottom right coordinates of the square
#top_left = (- 2390000, 1137000,)
#bottom_right = (-1640000, 589000)

top_left = (-2371000, 1105000)
bottom_right=(-1665000, 653000)

# Create a square polygon
square = create_square(top_left, bottom_right)

folder = "/data_1/ATM/data_1/papers/paper_georef"
file_name = "aoi_small"

# Save the polygon to a shapefile
save_shapefile(square, folder + "/" + file_name + ".shp")

print("Shapefile created successfully.")
