"""Load an image as a numpy array"""

# Library imports
import os
import numpy as np
import rasterio
import shutil
import warnings
from affine import Affine
from osgeo import gdal
from rasterio.transform import Affine as RasterioAffine
from typing import Union

# constant for default image path
DEFAULT_IMAGE_FLD = "/data/ATM/data_1/aerial/TMA/downloaded"
BACKUP_IMAGE_FLD = "/media/fdahle/d3f2d1f5-52c3-4464-9142-3ad7ab1ec06d/data_1/aerial/TMA/downloaded"

def load_image(image_id: str | bytes,
               image_path: str | None = None, image_type: str = "tif",
               driver: str = 'rasterio', return_transform: bool = False,
               catch: bool = False) -> Union[np.ndarray, tuple[np.ndarray,
                                                               Union[RasterioAffine, Affine]], None]:
    """
    Loads an image from the specified path and returns it as a numpy array.
    Args:
        image_id (str): Identifier for the image. Can be a path or just an ID.
        image_path (str, optional): Path to the image directory. Defaults to None.
        image_type (str, optional): Image file extension (without '.'). Defaults to 'tif'.
        driver (str, optional): Library to use for loading the image ('rasterio' or 'gdal').
            Defaults to 'rasterio'.
        return_transform (bool, optional): If True, returns the image transform alongside the image.
            Defaults to False.
        catch (bool, optional): If True, catches any exceptions and returns None. Defaults to False.
    Returns:
        numpy.ndarray: The loaded image as a numpy array.
        rasterio.transform.Affine or None: Image transform if 'return_transform' is True
            and the image is loaded successfully.
    Raises:
        FileNotFoundError: If the image file does not exist and `catch` is False.
    """

    # ignore warnings of files not being geo-referenced
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    # convert image_id to string
    image_id = image_id.decode() if isinstance(image_id, bytes) else image_id
    image_id = str(image_id)

    # construct absolute path for the image
    absolute_image_path = _create_absolute_path(image_id, image_path, image_type)

    try:
        img, transform = _read_image(absolute_image_path, driver)
    except (Exception,) as e:
        if catch:
            return (None, None) if return_transform else None
        else:
            raise e

    return (img, transform) if return_transform else img


def _create_absolute_path(image_id: str, fld: str, filetype: str) -> str:
    """ Constructs the absolute path for the image. """

    # path is already a file -> return immediately
    if os.path.isfile(image_id):
        return image_id

    # if "/" still in image_id, it's an invalid path
    if "/" in image_id:
        if os.path.isfile(image_id) is False:
            raise FileNotFoundError(f"No image could be found at {image_id}")

    # check image id and fld
    image_id = f"{image_id}.{filetype}" if "." not in image_id else image_id
    fld = fld if fld is not None else DEFAULT_IMAGE_FLD

    # create absolute image path
    absolute_image_path = os.path.join(fld, image_id)

    if absolute_image_path.startswith(DEFAULT_IMAGE_FLD) and not os.path.isfile(absolute_image_path):

        # Attempt to find the image in the backup folder
        backup_image_path = _create_absolute_path(image_id, BACKUP_IMAGE_FLD, filetype)

        print(f"Image not found at {absolute_image_path}. Attempting to load from backup at {backup_image_path}")

        os.makedirs(os.path.dirname(absolute_image_path), exist_ok=True)
        shutil.copy2(backup_image_path, absolute_image_path)

    # check if the path is valid
    if os.path.isfile(absolute_image_path) is False:
        raise FileNotFoundError(f"No image could be found at {absolute_image_path}")

    return absolute_image_path


def _read_image(absolute_image_path: str, driver: str) -> tuple[np.ndarray, Union[RasterioAffine, Affine]]:
    """ Reads an image using the specified driver. """

    # read with rasterio
    if driver == "rasterio":

        # read image
        try:
            ds = rasterio.open(absolute_image_path, 'r')
            img = ds.read()
        except Exception as e:
            print("Error loading image from ", absolute_image_path)
            raise e

        # get transform from image
        transform = ds.transform

        return (img[0], transform) if img.shape[0] == 1 else (img, transform)

    # read with gdal 
    elif driver == "gdal":

        # read image
        ds = gdal.Open(absolute_image_path)
        nbands = ds.RasterCount  # noqa: Spelling Error

        # different handling based on number of bands
        if nbands == 1:
            band = ds.GetRasterBand(1)
            img = band.ReadAsArray()
        else:
            img = np.zeros((ds.RasterYSize, ds.RasterXSize, nbands), dtype=np.uint8)
            for b in range(nbands):
                band = ds.GetRasterBand(b + 1)
                img[:, :, b] = band.ReadAsArray()

        # get transform of image
        gdal_transform = ds.GetGeoTransform()
        transform = Affine(gdal_transform[1], gdal_transform[2], gdal_transform[0],
                           gdal_transform[4], gdal_transform[5], gdal_transform[3])

        return img, transform

    else:
        raise ValueError("Unsupported driver. Choose 'rasterio' or 'gdal'.")


if __name__ == "__main__":
    img_id = "CA213733R0020"
    img = load_image(img_id)

    import src.display.display_images as di
    di.display_images(img)