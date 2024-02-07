import os
import numpy as np

DEFAULT_TRANSFORM_FLD = "C:/Users/Felix/Documents/GitHub/Antarctic_TMA/data/transformation"

def load_transform(image_id, transform_path=None):

    # define the absolute path to the file
    absolute_image_path = _create_absolute_path(image_id, transform_path)

    # load the matrix
    data_arr = np.loadtxt(absolute_image_path)

    # reshape to right format
    transform_matrix = data_arr.reshape([3, 3])

    return transform_matrix

def _create_absolute_path(image_id, fld):

    # path is already a file -> return immediately
    if os.path.isfile(image_id):
        return image_id

    # if "/" still in image_id, it's an invalid path
    if "/" in image_id:
        if os.path.isfile(image_id) is False:
            raise ValueError(f"No transformation matrix could be found at {image_id}")

    # check image id and fld
    image_id = f"{image_id}.txt" if "." not in image_id else image_id
    fld = fld if fld is not None else DEFAULT_TRANSFORM_FLD

    # create absolute image path
    absolute_transform_path = os.path.join(fld, image_id)

    # check if the path is valid
    assert os.path.isfile(absolute_transform_path), (f"No transformation matrix could be found at "
                                                     f"{absolute_transform_path}")

    return absolute_transform_path
