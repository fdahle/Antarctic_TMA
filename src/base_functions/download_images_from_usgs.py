import os

import wget
import shutil
import gzip
import ssl

import remove_usgs_logo as rul
import add_image_to_database as aitd

"""
download_images_from_usgs(ids, download_folder, add_to_db, url_images, verbose, catch):
With this function it is possible to download the images from the TMA archive and store them in a destination folder.
The images are downloaded in a zipped format, so after downloading they also need to be unpacked (and sometimes the
zipped file is not properly deleted).
INPUT:
    ids (List): A list of strings containing the ids of the images.
    download_folder (String): The folder in which the images should be stored.
    add_to_db (Boolean, True): If true the downloaded image will be stored in the db ("Images" and "Images_properties".
    url_images (String, None): The url where the images can be downloaded. If "None", these will be downloaded from the
        University of Minnesota
    verbose (Boolean, False): If true, the status of the operations are printed
    catch (Boolean, True): If true and somethings is going wrong, the operation will continue and not crash
        not required here, as nothing can go wrong, just here for the sake of continuity
OUTPUT:
    img_path (String): The path to which the images have been downloaded.
"""


def download_images_from_usgs(ids, download_folder, add_to_db=True, url_images=None, verbose=False, catch=True):

    # from here we can download the images
    if url_images is None:
        url_images = "http://data.pgc.umn.edu/aerial/usgs/tma/photos/high"

    # check if id is in a list -> if not put it in a list
    if isinstance(ids, list) is False:
        ids = [ids]

    # important as otherwise the download fails sometimes
    ssl._create_default_https_context = ssl._create_unverified_context # noqa

    # iterate all ids
    for img_id in ids:

        # get the image params
        tma = img_id[0:6]
        direction = img_id[6:9]
        roll = img_id[9:13]

        # define the url and download paths
        url = url_images + "/" + tma + "/" + tma + direction + "/" + tma + direction + roll + ".tif.gz"
        gz_path = download_folder + "/" + img_id + ".gz"
        img_path = download_folder + "/" + img_id + ".tif"

        if verbose:
            print(f"\rDownload {img_id} from {url}", end='')

        try:
            # download file
            wget.download(url, out=gz_path)

            # unpack file
            with gzip.open(gz_path) as f_in, open(img_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            # remove the gz file
            os.remove(gz_path)

            # if we have a new image we also want to remove the usgs logo and add it to the database
            if add_to_db:

                added = aitd.add_image_to_database(img_id, img_path, verbose=verbose, catch=catch)

                # remove the usgs logo
                removed = rul.remove_usgs_logo(img_path, verbose=verbose, catch=catch)

                # if one operation did not work
                if removed is False or added is False:
                    if verbose:
                        print(f"Something went wrong with adding {img_id} to the database")

                    # delete the image (because otherwise we have not good images)
                    os.remove(img_path)

                    if catch is False:
                        raise Exception

            if verbose:
                print(f"{img_id} successfully downloaded and added to the database")

            return img_path

        except (Exception,) as e:
            if catch:
                return None
            else:
                raise e
