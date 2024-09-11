"""download the reference elevation data of antarctica"""

import os
import sys
import tarfile
import wget

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
            def bar_progress(current, total):
                """
                create this bar_progress method which is invoked automatically from wget
                Args:
                    current: The current progress
                    total: The total progress
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

            # download file
            import time
            for _ in range(max_retries):
                try:
                    wget.download(url, out=file_path_gz, bar=bar_progress)
                    break  # Successful download, exit the loop
                except (Exception,) as e:
                    print("Download failed. Retrying in {} seconds...".format(retry_delay))
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

    print(f"Finished: download_rema_data ({tile})")

    return True


if __name__ == "__main__":

    # we want to download the tiles based on the mosaic of the rema data
    rema_shape_data = lsd.load_shape_data(PATH_REMA_SHP)

    tiles = [
        "43_06"
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
