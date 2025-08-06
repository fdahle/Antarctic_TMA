import gzip
import os
import shutil
import ssl
import wget
from markdown_it.common.entities import entities

PATH_DOWNLOAD_FLD = "/data/ATM/data_1/aerial/TMA/downloaded_child"
PATH_DOWNLOAD_FLD = "/media/fdahle/d3f2d1f5-52c3-4464-9142-3ad7ab1ec06d/data_1/aerial/TMA/downloaded"
USGS_URL = "http://data.pgc.umn.edu/aerial/usgs/tma/photos/high"


def download_image_from_usgs(image_id, download_folder=None, url_image=None,
                             overwrite=False, catch=True, verbose=False, pbar=None):
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

    print(f"Download {image_id} from USGS")

    # specify the download folder
    if download_folder is None:
        download_folder = PATH_DOWNLOAD_FLD

    # from here we can download the images
    if url_image is None:
        url_image = USGS_URL

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
        print(f"{image_id} is already downloaded")
        return

    try:
        # download file
        wget.download(url, out=gz_path)

        # unpack file
        with gzip.open(gz_path) as f_in, open(img_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        # remove the gz file
        os.remove(gz_path)

        print(f"{image_id} downloaded successfully")

        return True

    except (Exception,) as e:
        if catch:
            print(f"{image_id} downloaded failed")
            return False
        else:
            raise e

if __name__ == "__main__":

    # load the shape data
    import src.load.load_shape_data as lsd
    pth_photos = "/data/ATM/data_1/shapefiles/TMA_Photocenters/TMA_pts_20100927.shp"
    df = lsd.load_shape_data(pth_photos)

    mapping = {'L': '31L', 'V': '32V', 'R': '33R'}

    # for every row, for every letter in VIEWS, build the new code
    image_ids = [
        f"{eid[:6]}{mapping[letter]}{eid[6:]}"
        for eid, views in zip(df['ENTITY_ID'], df['VIEWS'])
        for letter in views
    ]

    # shuffle the image ids
    import random
    random.shuffle(image_ids)

    from tqdm import tqdm
    for image_id in tqdm(image_ids):
        download_image_from_usgs(image_id)
