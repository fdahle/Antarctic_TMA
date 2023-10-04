import gzip
import json
import os
import shutil
import ssl
import wget

import base.print_v as p


def download_image_from_usgs(image_id, download_folder=None, url_image=None,
                             overwrite=True, catch=True, verbose=False, pbar=None):
    """
    download_image_from_usgs(ids, download_folder, add_to_db, url_images, verbose, catch):
    With this function it is possible to download an image from the TMA archive and store it in
    the download folder. The image is downloaded in a zipped format, so after downloading it
    also need to be unpacked (and sometimes the zipped file is not properly deleted).
    Args:
        image_id (String): A string with the image_id of the image.
        download_folder (String, None): The folder in which the image should be stored.
            If "None", the default download path will be used
        url_image (String, None): The url where the images can be downloaded. If "None",
            these will be downloaded from the University of Minnesota
        overwrite (Boolean): If true and the image is already existing we're not downloading again
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar

    Returns:
        img_path (String): The path to which the images have been downloaded.
    """

    p.print_v(f"Download {image_id} from USGS", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # specify the download folder
    if download_folder is None:
        download_folder = json_data["path_folder_downloaded"]

    # from here we can download the images
    if url_image is None:
        url_image = json_data["url_usgs_download"]

    # important as otherwise the download fails sometimes
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa

    # get the image params
    tma = image_id[0:6]
    direction = image_id[6:9]
    roll = image_id[9:13]

    # define the url and download paths
    url = url_image + "/" + tma + "/" + tma + direction + "/" + tma + direction + roll + ".tif.gz"
    gz_path = download_folder + "/" + image_id + ".gz"
    img_path = download_folder + "/" + image_id + ".tif"

    # check if we already have this file and if we want to overwrite it
    if overwrite is False and os.path.isfile(img_path):
        p.print_v(f"{image_id} is already downloaded", verbose, "green", pbar)
        return

    try:
        # download file
        wget.download(url, out=gz_path)

        # unpack file
        with gzip.open(gz_path) as f_in, open(img_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        # remove the gz file
        os.remove(gz_path)

        p.print_v(f"{image_id} downloaded successfully", verbose, "green", pbar=pbar)

        return True

    except (Exception,) as e:
        if catch:
            p.print_v(f"{image_id} downloaded failed", verbose, "red", pbar=pbar)
            return False
        else:
            raise e
