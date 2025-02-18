import os
import rasterio
import warnings
from typing import LiteralString

# constant for default image path
DEFAULT_IMAGE_FLD = "/data/ATM/data_1/aerial/TMA/downloaded"
BACKUP_IMAGE_FLD = "/media/fdahle/d3f2d1f5-52c3-4464-9142-3ad7ab1ec06d/data_1/aerial/TMA/downloaded"


def load_image_shape(image_id: LiteralString | str | bytes,
                    image_path: str | None = None, image_type="tif") -> tuple[int, int]:

    # ignore warnings of files not being geo-referenced
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    # convert image_id to string
    image_id = image_id.decode() if isinstance(image_id, bytes) else image_id
    image_id = str(image_id)

    absolute_image_path = _create_absolute_path(image_id, image_path, image_type)

    with rasterio.open(absolute_image_path) as dataset:
        # Access metadata
        width = dataset.width
        height = dataset.height

    return height, width


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
        absolute_image_path = _create_absolute_path(image_id, BACKUP_IMAGE_FLD, filetype)

    # check if the path is valid
    if os.path.isfile(absolute_image_path) is False:
        raise FileNotFoundError(f"No image could be found at {absolute_image_path}")

    return absolute_image_path
