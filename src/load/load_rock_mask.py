""" load the rock-mask from quantarctica """

# Python imports
import os

import numpy as np
from fontTools.subset import subset

from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation
from shapely.geometry import box

import src.load.load_shape_data as lsd

# CONSTANTS
PATH_QUANTARTICA = "/data/ATM/data_1/quantarctica/Quantarctica3"  # noqa
PATH_ROCK_MASK = os.path.join(PATH_QUANTARTICA, "Geology/ADD/ADD_RockOutcrops_Landsat8.shp")

def load_rock_mask(bounds: list,
                   resolution: int | float | None = None,
                   mask_buffer: int = 0,
                   return_shapes=False) -> np.ndarray:
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

    # Check the resolution
    if return_shapes is False:
        if resolution is None:
            raise ValueError("resolution must be provided if return_shapes is False")
        if resolution < 1:
            raise Warning(f"Resolution is very low ({resolution})")

    # Load the shape data
    rock_shape_data = lsd.load_shape_data(PATH_ROCK_MASK,
                                          bounding_box=bounds, verbose=False)

    # Unpack bounds and calculate grid dimensions
    min_x, min_y, max_x, max_y = bounds

    # adjust bounds to resolution
    if resolution is not None:
        min_x = np.floor(min_x / resolution) * resolution
        min_y = np.floor(min_y / resolution) * resolution
        max_x = np.ceil(max_x / resolution) * resolution
        max_y = np.ceil(max_y / resolution) * resolution

    # Create a Shapely geometry for the bounding box
    bounding_box = (min_x, min_y, max_x, max_y)
    bbox_geom = box(*bounding_box)

    # select intersecting shapes
    subset_gdf = rock_shape_data[rock_shape_data.intersects(bbox_geom)]

    # Clip the geometries to the bounding box
    subset_gdf = subset_gdf.copy()  # To avoid potential SettingWithCopyWarning
    subset_gdf["geometry"] = subset_gdf["geometry"].intersection(bbox_geom)

    # remove all shapes with area over 1000000
    subset_gdf = subset_gdf[subset_gdf["geometry"].area < 500000]

    if return_shapes:

        subset_gdf = subset_gdf.copy()  # To avoid SettingWithCopyWarning
        subset_gdf["geometry"] = subset_gdf.intersection(bbox_geom)

        if subset_gdf.crs is None:
            subset_gdf.set_crs(epsg=3031, inplace=True)

        return subset_gdf

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
        dtype='uint8')

    # Apply mask buffer by dilating the mask (expanding the regions of 1s)
    if mask_buffer > 0:

        kernel = np.ones((mask_buffer, mask_buffer),
                         dtype=bool)
        rasterized = binary_dilation(rasterized, structure=kernel)

    return rasterized