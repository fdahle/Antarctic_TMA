import json
import os
import numpy as np
import rasterio
import warnings

from affine import Affine
from osgeo import gdal

import base.print_v as p


def load_image_from_file(image_id, image_path=None, image_type="tif",
                         driver="rasterio",
                         return_transform=False,
                         catch=True, verbose=False, pbar=None):
    """
    load_image_from_file(image_id, image_type, image_path, catch, verbose):
    This function loads an image from a specified path and returns it as a numpy array.
    Args:
        image_id (String): The image_id of the image that should be loaded.
        image_path (String, None): The path where the image is located. If this is None, the
            default aerial image path is used.
        image_type (String, "tif"): The type of image that should be loaded.
        driver (String, "rasterio"): Which package should be used for loading the images ("rasterio" or "gdal")
        return_transform (Boolean, False): If yes, the transform of the image is returned next to the image
        catch (Boolean, True): If true and something is going wrong, the operation will continue and not crash.
            In this case None is returned
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        img (np-array): The image loaded from the file
        transform (rasterio-transform): The transform of the image (describing the position, pixel-size, etc)
    """

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # already init variable
    transform = None

    # ignore warnings of files not being geo-referenced
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    # if image image_id is already a path, use that as an image path
    if "/" in image_id:
        absolute_image_path = image_id

    # otherwise the folder needs to be specified
    else:

        # if no image path is given we use default path
        if image_path is None:
            image_path = json_data["path_folder_downloaded"]

        # check if image path ends with '/' and add if not
        if image_path.endswith("/") is False:
            image_path = image_path + "/"

        # if the image type is already specified no need to add image type
        if len(image_id.split(".")) >= 2:
            image_type = ""
        else:
            image_type = "." + image_type

        # create absolute path
        absolute_image_path = image_path + image_id + image_type

    if catch is False:
        assert os.path.isfile(absolute_image_path), f"No image could be found at {absolute_image_path}"

    p.print_v(f"read {image_id} from {absolute_image_path}", verbose, pbar=pbar)

    try:
        if driver == "rasterio":
            # extract image from data
            ds = rasterio.open(absolute_image_path, 'r')
            img = ds.read()

            # if grayscale just return the first band
            if img.shape[0] == 1:
                img = img[0]

            # get the transform
            if return_transform:
                transform = ds.transform
        elif driver == "gdal":
            ds = gdal.Open(absolute_image_path)
            nbands = ds.RasterCount
            if nbands == 1:
                band = ds.GetRasterBand(1)
                img = band.ReadAsArray()
            else:
                img = np.zeros((ds.RasterYSize, ds.RasterXSize, nbands), dtype=np.uint8)
                for b in range(nbands):
                    band = ds.GetRasterBand(b + 1)
                    img[:, :, b] = band.ReadAsArray()
            if return_transform:
                gdal_transform = ds.GetGeoTransform()
                transform = Affine(gdal_transform[1],  # a
                                   gdal_transform[2],  # b
                                   gdal_transform[0],  # c
                                   gdal_transform[4],  # d
                                   gdal_transform[5],  # e
                                   gdal_transform[3])  # f
        else:
            raise ValueError("Unsupported driver. Choose 'rasterio' or 'gdal'.")

        if return_transform:
            return img, transform
        else:
            return img

    except (Exception,) as e:
        if catch:
            if return_transform:
                return None, None
            else:
                return None
        else:
            raise e


if __name__ == "__main__":
    img_id = "CA172031L0258"

    image = load_image_from_file(img_id)

    print(image.shape)
