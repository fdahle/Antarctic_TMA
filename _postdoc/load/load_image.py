"""Load an image as a numpy array"""

import warnings
import os

import numpy as np
import rasterio

DEFAULT_IMAGE_FLD = ""

def load_image(image: str,
               folder: str | None = None,
               return_transform: bool = False) -> (
        np.ndarray | tuple[np.ndarray, np.ndarray]):
    """
    Load a raster image as a NumPy array from a path or filename.

    Args:
        image: Filename or absolute path.
        folder: Folder to search if `image` is not a path. Defaults to env var IMAGE_FOLDER or DEFAULT_IMAGE_FLD.
        return_transform: If True, also return the affine transform.

    Returns:
        np.ndarray, or (np.ndarray, transform) if `return_transform` is True.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """

    # ignore warnings of files not being geo-referenced
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    # create absolute path for the image based on the provided image name and folder
    if os.path.isabs(image) or os.path.isfile(image):
        img_path = image
    else:
        if folder is None:
            folder = os.environ.get("IMAGE_FOLDER", DEFAULT_IMAGE_FLD)
        img_path = os.path.join(folder, image)

    # check if the image file exists
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"No image could be found at {img_path}")

    # read image using rasterio
    with rasterio.open(img_path, 'r') as ds:
        # read the first band (assuming single-band image)
        img = ds.read()

        # get transform
        transform = ds.transform

    # if the image has only one band, remove the first dimension
    if img.shape[0] == 1:
        img = img.squeeze(axis=0)

    if return_transform:
        return img, transform
    return img