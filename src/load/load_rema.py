"""Load REMA height data"""

# Library imports
import gc
import geopandas as gpd
import os
import numpy as np
import rasterio
import shapely
from rasterio import mask
from rasterio.merge import merge as merge_arrays
from rasterio.io import MemoryFile
from tqdm import tqdm

import src.other.misc.download_rema_data as drd

# Constants
DEFAULT_REMA_FLD = "/data/ATM/data_1/DEM/REMA/mosaic"
DEFAULT_REMA_SHP = "/data/ATM/data_1/DEM/REMA/overview/REMA_Mosaic_Index_v2"


def load_rema(bounds: tuple[float, float, float, float] or shapely.geometry.base.BaseGeometry,
              rema_shape_file: str = DEFAULT_REMA_SHP,
              rema_folder: str = DEFAULT_REMA_FLD,
              zoom_level: int = 10,
              auto_download=False,
              return_empty_rema: bool = False,
              return_transform=False) -> (np.ndarray | None, np.ndarray | None):
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
        auto_download (bool): If True, automatically downloads REMA data if not found locally.
            Defaults to False.
        return_empty_rema (bool): If True, returns an empty dataset instead of
            raising an exception when no images are found. Defaults to False.
    Returns:
        Optional[np.ndarray]: The cropped REMA image as a NumPy array or None if no images are found
            and return_empty_rema is True.

    """

    # check for correct zoom level
    if zoom_level not in [2, 10, 32]:
        raise ValueError("Zoom level not supported")

    # check if the output resolution fits the bounding box
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if width % zoom_level != 0 or height % zoom_level != 0:
        raise ValueError("Zoom level does not fit the bounding box")

    # check if we have the right folder
    if os.path.isdir(rema_folder) is False:
        raise FileNotFoundError(f"'{rema_folder}' is not a valid folder")

    # Convert bounds to shapely geometry if needed
    if not isinstance(bounds, shapely.geometry.base.BaseGeometry):
        bounds_geom = shapely.geometry.box(*bounds)
    else:
        bounds_geom = bounds
        bounds = bounds.bounds  # convert to tuple for later use

    # Load shapefile and filter using spatial index for speed
    shp_path = f"{rema_shape_file}_{zoom_level}m.shp"
    mosaic_data = gpd.read_file(shp_path)
    possible_matches_index = mosaic_data.sindex.intersection(bounds)
    possible_matches = mosaic_data.iloc[list(possible_matches_index)]
    intersects = possible_matches[possible_matches.intersects(bounds_geom)]
    tiles = intersects["dem_id"].tolist()


    if not tiles:
        if return_empty_rema:
            return (None, None) if return_transform else None
        raise FileNotFoundError("No REMA tiles found for the given bounds.")

    # Prepare tile folder
    zoom_folder = os.path.join(rema_folder, f"{zoom_level}m")
    memory_pairs = []

    try:

        pbar = tqdm(total=len(tiles), desc="Load tiles for REMA",
                    position=0, leave=True)

        for tile in tiles:
            tile_base = "_".join(tile.split("_")[:2])
            tile_path = os.path.join(zoom_folder, f"{tile_base}_{zoom_level}m.tif")

            if not os.path.exists(tile_path):
                if auto_download:
                    drd.download_rema_data(tile_base, zoom_level)
                elif return_empty_rema:
                    return (None, None) if return_transform else None
                else:
                    raise FileNotFoundError(f"Missing REMA tile: {tile_path}")

            with rasterio.open(tile_path) as src:
                cropped, transform = mask.mask(src, [bounds_geom], crop=True, all_touched=True)

                memfile = MemoryFile()
                dataset = memfile.open(driver="GTiff",
                                       height=cropped.shape[1],
                                       width=cropped.shape[2],
                                       count=1,
                                       dtype=cropped.dtype,
                                       crs=src.crs,
                                       transform=transform)
                dataset.write(cropped)
                dataset.close()
                memory_pairs.append((memfile, memfile.open()))  # reopen for merge

            pbar.update(1)

        # close progress bar
        pbar.set_postfix_str("- Finished -")
        pbar.close()

        datasets = [ds for _, ds in memory_pairs]
        merged, merged_transform = merge_arrays(datasets)
        merged = merged[0, :, :]  # first band

    finally:
        for memfile, dataset in memory_pairs:
            dataset.close()
            memfile.close()
        memory_pairs.clear()
        gc.collect()

    return (merged, merged_transform) if return_transform else merged
