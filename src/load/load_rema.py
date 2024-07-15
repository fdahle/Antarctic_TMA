"""Load REMA height data"""

# Library imports
import geopandas as gpd
import os
import numpy as np
import rasterio
import shapely
from rasterio import mask, merge
from typing import Tuple, Optional

# Constants
DEFAULT_REMA_FLD = "/data/ATM/data_1/DEM/REMA/mosaic"
DEFAULT_REMA_SHP = "/data/ATM/data_1/DEM/REMA/overview/REMA_Mosaic_Index_v2"


def load_rema(bounds: Tuple[float, float, float, float] or shapely.geometry.base.BaseGeometry,
              rema_shape_file: str = DEFAULT_REMA_SHP,
              rema_folder: str = DEFAULT_REMA_FLD,
              zoom_level: int = 10,
              return_empty_rema: bool = False) -> Optional[np.ndarray]:
    """
    Loads REMA height data from a specified folder, merges them based on a bounding box. It is
    required to prove the path to the shape file containing the mosaic tiles. Optionally returns
    an empty dataset if no images are found and return_empty_rema is True.
    Args:
        bounds (Tuple[float, float, float, float] or shapely.geometry.base.BaseGeometry):
            The bounding box to filter and crop REMA images as a tuple of (min_x, min_y, max_x, max_y)
            or a Shapely geometry object.
        rema_shape_file (str): The path to the shape file containing the mosaic tiles.
        Defaults to DEFAULT_REMA_SHP.
        rema_folder (str): The base folder where REMA images are stored. Defaults to DEFAULT_REMA_FLD.
        zoom_level (int): The zoom level of the REMA images to load. Defaults to 10.
        return_empty_rema (bool): If True, returns an empty dataset instead of
            raising an exception when no images are found. Defaults to False.
    Returns:
        Optional[np.ndarray]: The cropped REMA image as a NumPy array or None if no images are found
            and return_empty_rema is True.

    """

    # check for correct zoom level
    if zoom_level not in [2, 10, 32]:
        raise ValueError("Zoom level not supported")

    # check if we have the right folder
    if os.path.isdir(rema_folder) is False:
        raise FileNotFoundError(f"'{rema_folder}' is not a valid folder")

    # convert bounds to shapely polygon if not already polygon
    if isinstance(bounds, shapely.geometry.base.BaseGeometry) is False:
        bounds = shapely.geometry.box(*bounds)

    # load the mosaic tiles from shape file
    mosaic_data = gpd.read_file(rema_shape_file + f"_{str(zoom_level)}m.shp")

    # find which mosaic tiles are intersecting with the polygon
    intersects = mosaic_data.intersects(bounds)
    indices = intersects[intersects].index
    tiles = mosaic_data["dem_id"].iloc[indices].tolist()

    # adapt rema-folder to zoom level
    zoom_rema_folder = rema_folder + f"/{str(zoom_level)}m"

    # here we save all rema files we want to merge
    mosaic_files = []

    # iterate through all image files in the rema folder
    for tile in tiles:

        # we only need the first part of the tile
        tile_parts = tile.split("_")
        tile = tile_parts[0] + "_" + tile_parts[1]

        # create the file path
        rema_tile_path = zoom_rema_folder + "/" + tile + "_" + str(zoom_level) + "m.tif"

        # open the image file
        src = rasterio.open(rema_tile_path)

        # add the file to the list
        mosaic_files.append(src)

    # what happens if no rema files are found
    if len(mosaic_files) == 0:
        if return_empty_rema:
            return None, None
        else:
            raise ValueError("No satellite images were found")

    # merge the rema tiles
    merged, transform_merged = rasterio.merge.merge(mosaic_files)

    # close the connection to the mosaic files
    for file in mosaic_files:
        file.close()

    # get first band
    merged = merged[0, :, :]

    # crop merged files to the bounds
    with rasterio.io.MemoryFile() as mem_file:
        with mem_file.open(
                driver="GTiff",
                height=merged.shape[0],
                width=merged.shape[1],
                count=1,
                dtype=merged.dtype,
                transform=transform_merged,
        ) as dataset:
            dataset.write(merged, 1)
        with mem_file.open() as dataset:
            cropped, transform_cropped = rasterio.mask.mask(dataset, [bounds], crop=True)

    cropped = cropped[0, :, :]

    return cropped, transform_cropped
