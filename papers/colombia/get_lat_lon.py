import geopandas as gpd
from shapely.geometry import mapping
import pyproj
from functools import partial
from shapely.ops import transform

# Load the shapefile
shapefile_path = '/data/ATM/colombia/data/grids/grid_1000.shp'
gdf = gpd.read_file(shapefile_path)

# Reproject to WGS 84 (EPSG:4326)
gdf = gdf.to_crs(epsg=4326)

# Extract polygon coordinates in lat-long (WGS 84)
polygons_coords = []

# Helper function to convert decimal degrees to DMS format
def decimal_to_dms(decimal_coord):
    degrees = int(decimal_coord)
    minutes = int((abs(decimal_coord) - abs(degrees)) * 60)
    seconds = (abs(decimal_coord) - abs(degrees) - minutes / 60) * 3600
    return f"{abs(degrees)}°{minutes}′{seconds:.2f}″{'S' if decimal_coord < 0 else 'N'}" if degrees >= 0 else f"{abs(degrees)}°{minutes}′{seconds:.2f}″{'W' if decimal_coord < 0 else 'E'}"


for geom in gdf.geometry:
    if geom.type == 'Polygon':
        polygons_coords.append(list(geom.exterior.coords))
    elif geom.type == 'MultiPolygon':
        for poly in geom:
            polygons_coords.append(list(poly.exterior.coords))

# Print the coordinates
for idx, coords in enumerate(polygons_coords):
    print(f"Polygon {idx+1} coordinates (lat, long):")
    for coord in coords:
        lon_dms = decimal_to_dms(coord[0])
        lat_dms = decimal_to_dms(coord[1])
        print(f"({lat_dms}, {lon_dms})")  # coord[0] is longitude, coord[1] is latitude
