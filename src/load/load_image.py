import os
import numpy as np
import rasterio
import warnings

from affine import Affine
from osgeo import gdal

# Global constant for default image path
#DEFAULT_IMAGE_FLD = "C:/Users/Felix/Documents/GitHub/Antarctic_TMA/data/images"
DEFAULT_IMAGE_FLD = "/data_1/ATM/data_1/aerial/TMA/downloaded"

def load_image(image_id, image_path=None, image_type="tif",
               driver='rasterio', return_transform=False):
    """
    Loads an image from the specified path and returns it as a numpy array.

    Args:
        image_id (str): Identifier for the image. Can be a path or just an ID.
        image_path (str, optional): Path to the image directory. Defaults to None.
        image_type (str, optional): Image file extension (without '.'). Defaults to 'tif'.
        driver (str, optional): Library to use for loading the image ('rasterio' or 'gdal'). Defaults to 'rasterio'.
        return_transform (bool, optional): If True, returns the image transform alongside the image. Defaults to False.

    Returns:
        numpy.ndarray: The loaded image as a numpy array.
        rasterio.transform.Affine or None: Image transform if 'return_transform' is True
            and the image is loaded successfully.

    Raises:
        FileNotFoundError: If the image file does not exist and `catch` is False.

    Examples:
        Load an image with default settings:
        >>> image = load_image("example_image_id")
        
        Load an image with a specific path and type:
        # >>> image = load_image("example_image_id", "/path/to/images", "tif")
    """

    # ignore warnings of files not being geo-referenced
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    absolute_image_path = _create_absolute_path(image_id, image_path, image_type)

    img, transform = _read_image(absolute_image_path, driver)

    return (img, transform) if return_transform else img


def _create_absolute_path(image_id, fld, filetype):
    """ Constructs the absolute path for the image. """

    # path is already a file -> return immediately
    if os.path.isfile(image_id):
        return image_id

    # if "/" still in image_id, it's an invalid path
    if "/" in image_id:
        if os.path.isfile(image_id) is False:
            raise ValueError(f"No image could be found at {image_id}")

    # check image id and fld
    image_id = f"{image_id}.{filetype}" if "." not in image_id else image_id
    fld = fld if fld is not None else DEFAULT_IMAGE_FLD

    # create absolute image path
    absolute_image_path = os.path.join(fld, image_id)

    # check if the path is valid
    assert os.path.isfile(absolute_image_path), f"No image could be found at {absolute_image_path}"

    return absolute_image_path


def _read_image(absolute_image_path, driver):
    """ Reads an image using the specified driver. """

    # read with rasterio
    if driver == "rasterio":

        # read image
        ds = rasterio.open(absolute_image_path, 'r')
        img = ds.read()

        # get transform from image
        transform = ds.transform

        return (img[0], transform) if img.shape[0] == 1 else (img, transform)

    # read with gdal 
    elif driver == "gdal":

        # read image
        ds = gdal.Open(absolute_image_path)
        nbands = ds.RasterCount

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
