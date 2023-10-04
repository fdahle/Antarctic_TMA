import json
import os
import tarfile
import wget

from tqdm import tqdm

import base.print_v as p


def download_rema_data(tile, zoom_level, destination_folder=None, download_url=None,
                       overwrite=False, catch=False, verbose=False, pbar=None):

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
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar

    Returns:
        status (Boolean): True, if download was successful
    """

    p.print_v(f"Start: download_rema_data ({tile})", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if destination_folder is None:
        destination_folder = json_data["path_folder_rema"] + str(zoom_level) + "m"

    # set download path
    if download_url is None:
        if zoom_level == 2:
            download_url = json_data["url_ugc_rema_data_2"]
        elif zoom_level == 10:
            download_url = json_data["url_ugc_rema_data_10"]
        elif zoom_level == 32:
            download_url = json_data["url_ugc_rema_data_32"]

    # get all download and file paths
    url = download_url + tile + "/" + tile + "_" + str(zoom_level) + "m_v2.0.tar.gz"
    file_path_gz = destination_folder + "/" + tile + "_" + str(zoom_level) + "m.tar.gz"
    file_path_dem = destination_folder + "/" + tile + "_" + str(zoom_level) + "m_v2.0_dem.tif"
    file_path_tif = destination_folder + "/" + tile + "_" + str(zoom_level) + "m.tif"

    # check if the file is already existing
    if os.path.isfile(file_path_tif) and overwrite is False:
        p.print_v(f"{tile} is already existing", verbose, "green", pbar=pbar)
        return True

    try:
        import sys
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
            except Exception as e:
                print("Download failed. Retrying in {} seconds...".format(retry_delay))
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
            p.print_v(f"Failed: download_rema_data ({tile})", verbose=verbose, pbar=pbar)
            return False
        else:
            raise e

    p.print_v(f"Finished: download_rema_data ({tile})", verbose=verbose, pbar=pbar)

    return True


if __name__ == "__main__":

    import base.load_shape_data as lsd

    path_rema_shape = "/data_1/ATM/data_1/shapefiles/REMA/REMA_Tile_Index_Rel1_1.shp"

    # we want to download the tiles based on the mosaic of the rema data
    rema_shape_data = lsd.load_shape_data(path_rema_shape)

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

    tiles=["25_15", "25_14"]

    _zoom_level = 32

    # iterate rows to download the equivalent satellite images
    for index, row in rema_shape_data.iterrows():

        # get the name (based on the tile
        _tile = row["tile"]

        if _tile not in tiles:
            continue

        # download the data
        download_rema_data(_tile, _zoom_level, catch=False, verbose=True)
