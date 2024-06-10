"""Exports a given image to a TIFF file"""

import cv2
import os
import numpy as np


def export_tiff(img: np.ndarray, output_path: str, overwrite: bool = False) -> None:
    """
    Exports a given image to a TIFF file by using OpenCV's imwrite function. If the file already
    exists, the function raises a FileExistsError unless the overwrite parameter is set to True.

    Args:
        img (np.ndarray): The image to export.
        output_path (str): The file path where the image should be saved.
        overwrite (bool, optional): If True, allows overwriting an existing file. Defaults to False.
    Returns:
        None
    """

    # Check if file exists and raise error if overwrite is False
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"The file {output_path} already exists. Set 'overwrite' to True to overwrite the file.")

    # Export the image to a TIFF file
    cv2.imwrite(output_path, img)
    print(f"Image successfully exported to {output_path}.")
