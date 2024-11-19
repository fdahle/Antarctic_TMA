"""Exports a given image to a TIFF file"""

import os

import rasterio
import numpy as np

from rasterio.transform import Affine
from typing import LiteralString


def export_tiff(img: np.ndarray, output_path: LiteralString | str | bytes,
                overwrite: bool = False,
                transform: Affine | np.ndarray | None = None,
                use_lzw: bool = False,
                crs: str = 'EPSG:3031', no_data=None) -> None:
    """
    Exports a given image to a TIFF file by using Rasterio's write function. If the file already
    exists, the function raises a FileExistsError unless the overwrite parameter is set to True.

    Args:
        img (np.ndarray): The image to export.
        output_path (str): The file path where the image should be saved.
        overwrite (bool, optional): If True, allows overwriting an existing file. Defaults to False.
        transform (rasterio.transform.Affine, optional): The transform to apply to the image. Defaults to None.
        use_lzw (bool, optional): If True, applies LZW compression to the TIFF file. Defaults to False.
        crs (str, optional): The coordinate reference system to apply to the image. Defaults to 'EPSG:4326'.
        no_data (optional): The value to use for no data pixels. Defaults to None.
    Returns:
        None
    """

    # Check if file exists and raise error if overwrite is False
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"The file {output_path} already exists. "
                              f"Set 'overwrite' to True to overwrite the file.")

    # convert bool numpy array to uint8
    if img.dtype == bool:
        img = img.astype(np.uint8)

    # Convert transform to Affine if necessary
    if isinstance(transform, np.ndarray):

        transform = Affine(*transform.flatten()[:6])

    # Define metadata for the TIFF file
    metadata = {
        'driver': 'GTiff',
        'dtype': img.dtype,
        'width': img.shape[1],
        'height': img.shape[0],
        'count': 1 if len(img.shape) == 2 else img.shape[2],  # Handle both single band and multi-band images
        'crs': crs if transform else None,
        'transform': transform if transform else Affine.identity()
    }

    if use_lzw:
        metadata['compress'] = 'lzw'

    if no_data is not None:
        metadata['nodata'] = no_data

    # Export the image to a TIFF file using Rasterio
    with rasterio.open(output_path, 'w', **metadata) as dst:
        if len(img.shape) == 2:  # Single band
            dst.write(img, 1)
        else:  # Multi-band
            for i in range(img.shape[2]):
                dst.write(img[:, :, i], i + 1)
