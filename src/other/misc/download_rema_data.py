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


def download_rema_data(tile, zoom_level, destination_folder=None, download_url=None,
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

    # define subtiles for zoom level 2

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
            def bar_progress(current, total, width=80):
                progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
                # Don't use print() as it will print in new line every time.
                sys.stdout.write("\r" + progress_message)
                sys.stdout.flush()

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

    tiles = ["48_06", "47_06", "47_05", "47_04", "46_07", "46_06", "46_05", "46_04",
             "45_07", "45_06", "45_05", "45_04", "44_08", "44_07", "44_06", "44_05",
             "44_04", "43_09", "43_08", "43_07", "43_06", "43_05", "42_11", "42_10",
             "42_09", "42_08", "42_07", "42_06", "42_05", "41_13", "41_12", "41_11",
             "41_10", "41_09", "41_08", "41_07", "41_06", "40_15", "40_14", "40_13",
             "40_12", "40_11", "40_10", "40_09", "40_08", "40_07", "39_17", "39_16",
             "39_15", "39_14", "39_13", "39_12", "39_11", "39_10", "39_09", "39_08",
             "39_07", "38_17", "38_16", "38_15", "38_14", "38_13", "38_12", "38_11",
             "38_10", "38_09", "38_08", "37_17", "37_16", "37_15", "37_14", "37_13",
             "37_12", "37_11", "37_10", "37_09", "37_08", "36_17", "36_16", "36_15",
             "36_14", "36_13", "36_12", "36_11", "36_10", "36_09", "35_17", "35_16",
             "35_15", "35_14", "35_13", "35_12", "35_11", "34_17", "34_16", "34_15",
             "34_14", "34_13", "34_12", "34_11"]

    tiles = ["38_11"]

    _zoom_level = 10

    # iterate rows to download the equivalent satellite images
    for index, row in rema_shape_data.iterrows():

        # get the name (based on the tile
        _tile = row["tile"]

        if _tile not in tiles:
            continue

        # download the data
        download_rema_data(_tile, _zoom_level, catch=False)
