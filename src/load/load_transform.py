# Package imports
import os
import numpy as np
from typing import Optional

# Constants
DEFAULT_TRANSFORM_FLD = "C:/Users/Felix/Documents/GitHub/Antarctic_TMA/data/transformation"


def load_transform(transform_id: str, transform_fld: Optional[str] = None) -> np.ndarray:
    """
    Loads a transformation matrix from a specified file.
    Args:
        transform_id: A string identifier for the transform. This can be a path to a file or just an ID.
        transform_fld: An optional path to a directory containing the transformation files.
                        If None, uses the default transformation folder.
    Returns:
        A 3x3 numpy array representing the transformation matrix.
    """

    # define the absolute path to the transform file
    absolute_transform_path = _create_absolute_path(transform_id, transform_fld)

    # load the matrix
    data_arr = np.loadtxt(absolute_transform_path)

    # reshape to right format
    transform_matrix = data_arr.reshape([3, 3])

    return transform_matrix


def _create_absolute_path(transform_id: str, fld: Optional[str] = None) -> str:
    """
    Create an absolute path for a transformation matrix file.
    Args:
        transform_id (str): The ID or filename of the transformation matrix.
        fld (str, optional): The folder path where the transformation matrix file is located.
            If not provided, the default folder path is used.
    Returns:
        str: The absolute path to the transformation matrix file.
    Raises:
        FileNotFoundError: If the transformation matrix file cannot be found at the specified path.
    """

    # path is already a file -> return immediately
    if os.path.isfile(transform_id):
        return transform_id

    # if "/" still in image_id, it's an invalid path
    if "/" in transform_id:
        if os.path.isfile(transform_id) is False:
            raise FileNotFoundError(f"No transformation matrix could be found at {transform_id}")

    # check image id and fld
    transform_id = f"{transform_id}.txt" if "." not in transform_id else transform_id
    fld = fld if fld is not None else DEFAULT_TRANSFORM_FLD

    # create absolute image path
    absolute_transform_path = os.path.join(fld, transform_id)

    # check if the path is valid
    if os.path.isfile(absolute_transform_path) is False:
        raise FileNotFoundError(f"No transformation matrix could be found at {absolute_transform_path}")

    return absolute_transform_path
