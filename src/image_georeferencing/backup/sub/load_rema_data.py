import geopandas as gpd
import json
import os
import rasterio

from rasterio.merge import merge
from rasterio.mask import mask
from shapely import geometry

import other.download_rema_data as drd

import base.print_v as p


def load_rema_data(bounds, zoom_level=None, rema_folder=None, mosaic_shp_path=None,
                   catch=True, verbose=False, pbar=None):

    """load_rema_data(bounds, rema_folder, zoom_level, catch):
    With this function it is possible to download the reference elevation data of antarctica.
    The image is downloaded in a zipped format, so after downloading it also need to be
    unpacked (and sometimes the zipped file is not properly deleted).
    Args:
        bounds (String): From which area do we want to have elevation data
            [min_x, max_y, min_y, max_y]
        rema_folder (String): Where is the rema data stored
        zoom_level (String): The zoom level we want to get data for (2, 10, 32)
        mosaic_shp_path (String)
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        cropped (numpy-arr): The numpy array with the elevation data for bounds
    """

    p.print_v(f"Start: load_rema_data (bounds: {bounds})", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder[:-4] + "/params.json") as j_file:
        json_data = json.load(j_file)

    # set default zoom level if not specified
    if zoom_level is None:
        zoom_level = json_data["rema_zoom_level"]

    # set default destination folder
    if rema_folder is None:
        if zoom_level == 2:
            rema_folder = json_data["path_folder_rema_data_2"]
        elif zoom_level == 10:
            rema_folder = json_data["path_folder_rema_data_10"]
        elif zoom_level == 32:
            rema_folder = json_data["path_folder_rema_data_32"]

    # get mosaic shp path
    if zoom_level == 2:
        mosaic_shp_path = json_data["path_file_rema_mosaic_shp_2"]
    elif zoom_level == 10:
        mosaic_shp_path = json_data["path_file_rema_mosaic_shp_10"]
    elif zoom_level == 32:
        mosaic_shp_path = json_data["path_file_rema_mosaic_shp_32"]

    # check zoom level and folder
    assert zoom_level in [2, 10, 32]
    assert os.path.isdir(rema_folder)

    # define size of polygon & create polygon
    min_x = bounds[0]
    max_x = bounds[2]
    min_y = bounds[1]
    max_y = bounds[3]
    b_poly = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y], [min_x, min_y]]
    b_poly = geometry.Polygon([[po[0], po[1]] for po in b_poly])

    # load the mosaic tiles
    mosaic_data = gpd.read_file(mosaic_shp_path)  # noqa

    # find which mosaic tiles are intersecting with the polygon
    intersects = mosaic_data.intersects(b_poly)
    indices = intersects[intersects].index
    tiles = mosaic_data["dem_id"].iloc[indices].tolist()

    # iterate to download & open files
    mosaic_files = []
    for tile in tiles:

        # we only need the first part of the tile
        tile_parts = tile.split("_")
        tile = tile_parts[0] + "_" + tile_parts[1]

        file_path_tif = rema_folder + "/" + tile + "_" + str(zoom_level) + "m.tif"

        # check if tile is already downloaded
        if os.path.isfile(file_path_tif) is False:
            if catch:
                p.print_v(f"Tile {tile} is missing")
                continue
            else:
                raise ValueError(f"Tile '{tile}' is missing")

        src = rasterio.open(file_path_tif)
        mosaic_files.append(src)

    if len(mosaic_files) == 0:

        if catch:
            p.print_v("Failed: load_rema_data (No tiles were found for rema data)", verbose=verbose, pbar=pbar)
            return None
        else:
            raise ValueError("No tiles were found for rema data")

    # merge the tiles
    merged, merged_trans = merge(mosaic_files)

    # get first band
    merged = merged[0, :, :]

    with rasterio.io.MemoryFile() as mem_file:
        with mem_file.open(
                driver="GTiff",
                height=merged.shape[0],
                width=merged.shape[1],
                count=1,
                dtype=merged.dtype,
                transform=merged_trans,
        ) as dataset:
            dataset.write(merged, 1)
        with mem_file.open() as dataset:
            cropped, cropped_trans = mask(dataset, [b_poly], crop=True)

    cropped = cropped[0, :, :]

    p.print_v(f"Finished: load_rema_data", verbose=verbose, pbar=pbar)

    return cropped
