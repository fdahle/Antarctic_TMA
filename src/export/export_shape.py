import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def export_shape(np_array, output_file):

    # Extract x and y coordinates
    geometry = [Point(xy) for xy in np_array[:, :2]]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(np_array[:, 2:], columns=['Attribute_{}'.format(i) for i in range(np_array.shape[1] - 2)],
                           geometry=geometry)

    # Export to shapefile
    gdf.to_file(output_file)