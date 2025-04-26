"""download the reference elevation data of antarctica"""

import os
import rasterio
import requests
import sys
import tarfile
import time 
from rasterio.merge import merge
from glob import glob
from tqdm import tqdm

import src.load.load_shape_data as lsd

# constants
PATH_REMA_FLD = "/data/ATM/data_1/DEM/REMA/mosaic/"
PATH_REMA_SHP = "/data/ATM/data_1/DEM/REMA/Tiles/REMA_Tile_Index_Rel1_1.shp"
URL_REMA_2 = "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/2m/"
URL_REMA_10 = "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/10m/"
URL_REMA_32 = "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/32m/"


def download_rema_data(tile, zoom_level,
                       destination_folder=None, download_url=None,
                       overwrite=False, catch=False):
    """download_rema_data(tile, zoom_level, destination_folder, pbar, catch):
    With this function it is possible to download the reference elevation data of antarctica.
    The image is downloaded in a zipped format, so after downloading it also need to be
    unpacked (and sometimes the zipped file is not properly deleted).
    Args:
        tile (String): The image_id of the tile we want to download
        zoom_level (String): The zoom level we want to download (2, 10, 32)
        destination_folder (String): Where do we want to download the file
        download_url (String): From where are we downloading the data
        overwrite (Boolean, False): If true, the downloaded date will be overwritten
        catch (Boolean, True): If true and something is going wrong, the operation will continue
            and not crash

    Returns:
        status (Boolean): True, if download was successful
    """

    print(f"Start: download_rema_data ({tile})")

    # save current stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # restore original stdout and stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    # set destination folder
    if destination_folder is None:
        destination_folder = PATH_REMA_FLD + str(zoom_level) + "m"

    # set download path
    if download_url is None:
        if zoom_level == 2:
            download_url = URL_REMA_2
        elif zoom_level == 10:
            download_url = URL_REMA_10
        elif zoom_level == 32:
            download_url = URL_REMA_32

    # define sub_tiles for zoom level 2

    if zoom_level == 2:
        sub_tiles = ["_1_1", "_1_2", "_2_1", "_2_2"]
    else:
        sub_tiles = [""]

    for sub_tile in sub_tiles:

        if sub_tile != "":
            print("  Downloading sub-tile: ", sub_tile)

        # get all download and file paths
        url = download_url + tile + "/" + tile + sub_tile + "_" + str(zoom_level) + "m_v2.0.tar.gz"
        file_path_gz = destination_folder + "/" + tile + sub_tile + "_" + str(zoom_level) + "m.tar.gz"
        file_path_dem = destination_folder + "/" + tile + sub_tile + "_" + str(zoom_level) + "m_v2.0_dem.tif"
        file_path_tif = destination_folder + "/" + tile + sub_tile + "_" + str(zoom_level) + "m.tif"

        # check if the file is already existing
        if os.path.isfile(file_path_tif) and overwrite is False:
            print(f"{tile} {sub_tile} is already existing")
            continue

        try:
            # create this bar_progress method which is invoked automatically from wget
            def bar_progress(current, total, ratio):
                """
                create this bar_progress method which is invoked automatically from wget
                Args:
                    current: The current progress
                    total: The total progress
                    ratio: The ratio of current to total (not used here)
                """

                progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
                # Don't use print() as it will print in new line every time.
                if hasattr(sys.stdout, 'fileno'):
                    sys.stdout.write("\r" + progress_message)
                    sys.stdout.flush()
                else:
                    # If sys.stdout doesn't have fileno, use print instead
                    print(progress_message)

            max_retries = 5
            retry_delay = 5  # seconds

            for _ in range(max_retries):
                try:
                    response = requests.get(url, stream=True, timeout=30)
                    if response.status_code == 404:
                        print(f"{tile}{sub_tile} not found on server (404). Skipping.")
                        continue
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 1024 * 1024  # 1 MB

                    tqdm_bar = tqdm(total=total_size, unit='MB', unit_scale=True, desc=f"Downloading {tile}{sub_tile}",
                                    ncols=80)

                    with open(file_path_gz, 'wb') as f:
                        for data in response.iter_content(block_size):
                            f.write(data)
                            tqdm_bar.update(len(data))
                    tqdm_bar.close()
                    break  # Successful download
                except Exception as e:
                    print(f"Download failed. Retrying in {retry_delay} seconds...")
                    print(e)
                    time.sleep(retry_delay)

            if os.path.isfile(file_path_gz) is False:
                return

            # unpack file
            tar = tarfile.open(file_path_gz)
            tar.extract(file_path_dem.split("/")[-1], path=destination_folder)
            tar.close()
            os.rename(file_path_dem, file_path_tif)
            os.remove(file_path_gz)

        except (Exception,) as e:
            if catch:
                print(f"Failed: download_rema_data ({tile})")
                return False
            else:
                raise e

    # for sub tiles we need to merge the files into a single file
    if zoom_level == 2:
        # Search all the individual sub-tiles just downloaded
        pattern = os.path.join(destination_folder, f"{tile}_*_2m.tif")
        sub_tile_paths = sorted(glob(pattern))

        if not sub_tile_paths:
            print(f"Warning: no sub-tiles found for {tile}")
            return False

        print(f"Found {len(sub_tile_paths)} sub-tile(s) for {tile}, proceeding with merge.")

        # Open only available sub-tile datasets
        src_files_to_mosaic = []
        for fp in sub_tile_paths:
            try:
                src_files_to_mosaic.append(rasterio.open(fp))
            except rasterio.errors.RasterioIOError:
                print(f"Could not open {fp}, skipping.")

        if not src_files_to_mosaic:
            print(f"Warning: no valid sub-tile datasets found for {tile}")
            return False

        # Merge them
        mosaic, out_trans = merge(src_files_to_mosaic)

        # Write merged file
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw",
            "BIGTIFF": "YES",
        })

        merged_tile_path = os.path.join(destination_folder, f"{tile}_2m.tif")
        with rasterio.open(merged_tile_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        for src in src_files_to_mosaic:
            src.close()

        for f in sub_tile_paths:
            os.remove(f)

    # restore original stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    print(f"Finished: download_rema_data ({tile})")

    return True


if __name__ == "__main__":

    # we want to download the tiles based on the mosaic of the rema data
    rema_shape_data = lsd.load_shape_data(PATH_REMA_SHP)

    tiles = [
        "42_06"
    ]

    _zoom_level = 10

    # iterate rows to download the equivalent satellite images
    for index, row in rema_shape_data.iterrows():

        # get the name (based on the tile
        _tile = row["tile"]

        if _tile not in tiles:
            continue

        # download the data
        download_rema_data(_tile, _zoom_level, catch=False)
