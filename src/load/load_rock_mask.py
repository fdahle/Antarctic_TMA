# Python imports
import os

from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

import src.load.load_shape_data as lsd

# CONSTANTS
PATH_QUANTARTICA = "/data/ATM/data_1/quantarctica/Quantarctica3"
PATH_ROCK_MASK = os.path.join(PATH_QUANTARTICA, "Basemap/ADD_Rock_outcrop_?QUALITY?_res_polygon.shp")


def load_rock_mask(bounds, resolution, quality: str = "high"):

    # Check the quality
    if quality not in ["low", "medium", "high"]:
        raise ValueError("quality must be one of 'low', 'medium', 'high'")

    # Check the resolution
    if resolution < 1:
        raise Warning(f"Resolution is very low ({resolution})")

    # update path with quality
    path = PATH_ROCK_MASK.replace("?QUALITY?", quality)

    # Load the shape data
    rock_shape_data = lsd.load_shape_data(path)

    # Unpack bounds and calculate grid dimensions
    minx, min_y, max_x, maxy = bounds

    # Create a Shapely geometry for the bounding box
    bounding_box = (minx, min_y, max_x, maxy)
    bbox_geom = box(*bounding_box)

    # Clip the GeoDataFrame using the bounding box
    subset_gdf = rock_shape_data[rock_shape_data.intersects(bbox_geom)]

    # Calculate the bounds of the subset (to fit within the bbox)
    minx, min_y, max_x, maxy = bbox_geom.bounds

    print("minx, min_y, max_x, maxy", minx, min_y, max_x, maxy)

    # Calculate the number of rows and columns in the output raster
    nrows = int((maxy - min_y) / resolution)
    ncols = int((max_x - minx) / resolution)

    print("nrows, ncols", nrows, ncols)
    print("resolution", resolution)

    # Define the affine transform for the raster based on the bounding box
    transform = from_origin(minx, maxy, resolution, resolution)

    print("transform", transform)

    # Initialize the shape for the raster
    out_shape = (nrows, ncols)

    # Rasterize the subset of polygons
    rasterized = rasterize(
        [(geom, 1) for geom in subset_gdf.geometry],  # (geometry, value) tuples
        out_shape=out_shape,
        transform=transform,
        fill=0,  # Fill value for areas outside the polygons
        dtype='uint8'
    )

    return rasterized
