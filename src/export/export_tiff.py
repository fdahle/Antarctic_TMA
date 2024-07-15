"""Exports a given image to a TIFF file"""

import rasterio
from rasterio.transform import Affine
import os
import numpy as np


def export_tiff(img: np.ndarray, output_path: str,
                overwrite: bool = False,
                transform: Affine = None,
                crs: str = 'EPSG:3031', no_data=-9999) -> None:
    """
    Exports a given image to a TIFF file by using Rasterio's write function. If the file already
    exists, the function raises a FileExistsError unless the overwrite parameter is set to True.

    Args:
        img (np.ndarray): The image to export.
        output_path (str): The file path where the image should be saved.
        overwrite (bool, optional): If True, allows overwriting an existing file. Defaults to False.
        transform (rasterio.transform.Affine, optional): The transform to apply to the image. Defaults to None.
        crs (str, optional): The coordinate reference system to apply to the image. Defaults to 'EPSG:4326'.
    Returns:
        None
    """

    # Check if file exists and raise error if overwrite is False
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"The file {output_path} already exists. Set 'overwrite' to True to overwrite the file.")

    print(output_path)
    print(img.shape)

    # Define metadata for the TIFF file
    metadata = {
        'driver': 'GTiff',
        'dtype': img.dtype,
        'nodata': no_data,
        'width': img.shape[1],
        'height': img.shape[0],
        'count': 1 if len(img.shape) == 2 else img.shape[2],  # Handle both single band and multi-band images
        'crs': crs if transform else None,
        'transform': transform if transform else Affine.identity()
    }

    # Export the image to a TIFF file using Rasterio
    with rasterio.open(output_path, 'w', **metadata) as dst:
        if len(img.shape) == 2:  # Single band
            dst.write(img, 1)
        else:  # Multi-band
            for i in range(img.shape[2]):
                dst.write(img[:, :, i], i + 1)

    print(f"Image successfully exported to {output_path}.")
