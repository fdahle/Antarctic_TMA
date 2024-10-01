""" load the rock-mask from quantarctica """

# Python imports
import os

from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

import src.load.load_shape_data as lsd

# CONSTANTS
PATH_QUANTARTICA = "/data/ATM/data_1/quantarctica/Quantarctica3"  # noqa
PATH_ROCK_MASK = os.path.join(PATH_QUANTARTICA, "Basemap/ADD_Rock_outcrop_?QUALITY?_res_polygon.shp")


def load_rock_mask(bounds: list, resolution: int | float, quality: str = "high"):
    """
    Use the absolute bounds to load the rock mask from Quantarctica. The shape file
    will be limited to the bounds and rasterized with the given resolution.
    Args:
        bounds (list): The absolute bounds of the area to load the rock mask for.
            Is a list of the form [minx, min_y, max_x, maxy].
        resolution (int | float): The resolution of the rock mask in meters.
        quality (str, optional): Which rock mask shape file to use. Defaults to "high".

    Returns:

    """

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

    # Calculate the number of rows and columns in the output raster
    n_rows = int((maxy - min_y) / resolution)
    n_cols = int((max_x - minx) / resolution)

    # Define the affine transform for the raster based on the bounding box
    transform = from_origin(minx, maxy, resolution, resolution)

    # Initialize the shape for the raster
    out_shape = (n_rows, n_cols)

    # Rasterize the subset of polygons
    rasterized = rasterize(
        [(geom, 1) for geom in subset_gdf.geometry],  # (geometry, value) tuples
        out_shape=out_shape,
        transform=transform,
        fill=0,  # Fill value for areas outside the polygons
        dtype='uint8'
    )

    return rasterized
