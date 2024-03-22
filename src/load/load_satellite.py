import os
import numpy as np
import rasterio
import shapely

from rasterio import Affine
from rasterio import mask, merge
from typing import Optional, Tuple

DEFAULT_SAT_FLD = "/data_1/ATM/data_1/satellite"


def load_satellite(bounds: Tuple[float, float, float, float] or shapely.geometry.base.BaseGeometry,
                   sat_folder: str = DEFAULT_SAT_FLD,
                   satellite_type: str = "sentinel2",
                   satellite_crs: int = 3031,
                   month: int = 0,
                   return_empty_sat: bool = False) -> (
        Tuple)[Optional[np.ndarray], Optional[Affine]]:
    """
    Loads satellite images from a specified folder, merges them based on a bounding box,
    and crops the merged image to the given bounds. Optionally returns an empty dataset
    if no images are found and return_empty_sat is True.

    Args:
        bounds (Tuple[float, float, float, float] or shapely.geometry.base.BaseGeometry):
            The bounding box to filter and crop satellite images as a tuple of (min_x, min_y, max_x, max_y)
            or a Shapely geometry object.
        sat_folder (str): The base folder where satellite images are stored. Defaults to DEFAULT_SAT_FLD.
        satellite_type (str): The type of satellite data to load, e.g., "sentinel2". Defaults to "sentinel2".
        satellite_crs (int): The coordinate reference system (CRS) code to match satellite images.
            Defaults to 3031.
        month (int): The month for which satellite images are to be loaded. Use 0 for a composite over the
            complete year.
        return_empty_sat (bool): If True, returns an empty dataset instead of raising an exception when no images
            are found. Defaults to False.

    Returns:
        Tuple[Optional[np.ndarray], Optional[Affine]]: A tuple containing the cropped satellite image as a NumPy array
        and the affine transform of the cropped image, or None values if no images are found and return_empty_sat is
            True.

    Raises:
        ValueError: If the specified satellite type is not supported or no matching satellite images were found
        and return_empty_sat is False.
    """

    # adapt path to load from a certain satellite type
    if satellite_type == "sentinel2":
        sat_folder = sat_folder + "/sentinel_2"
    else:
        raise ValueError(f"Satellite type '{satellite_type}' not supported yet'")

    # conversion of month to monthly path strings
    month_strings = [
        "0_complete", "1_jan", "2_feb", "3_mar", "4_apr", "5_may", "6_jun",
        "7_jul", "8_aug", "9_sep", "10_oct", "11_nov", "12_dec"
    ]

    # adapt path to load for a certain month (0 means composite over complete year)
    sat_folder = sat_folder + "/" + month_strings[month]

    # last check if we have the right folder
    if os.path.isdir(sat_folder) is False:
        raise FileNotFoundError(f"'{sat_folder}' is not a valid folder")

    # convert bounds to shapely polygon if not already polygon
    if isinstance(bounds, shapely.geometry.base.BaseGeometry) is False:
        bounds = shapely.geometry.box(*bounds)

    # here we save all satellite files we want to merge
    mosaic_files = []

    # iterate through all image files in the satellite folder
    for file in os.listdir(sat_folder):
        if file.endswith(".tif"):

            # open the image file
            src = rasterio.open(sat_folder + "/" + file)

            # we only want the satellite images with the same crs code
            crs_code = int(src.crs['init'].split(":")[1])
            if crs_code != satellite_crs:
                src.close()
                continue

            # get bounding box of satellite image as shapely polygon
            sat_bounds = shapely.geometry.box(*src.bounds)

            # if the bounding boxes do not intersect, we can skip this raster file
            if bounds.intersects(sat_bounds) is False:
                src.close()
                continue

            mosaic_files.append(src)

    if len(mosaic_files) == 0:
        if return_empty_sat:
            return None, None,
        else:
            raise ValueError("No satellite images were found")

    # merge the satellite files
    merged, transform_merged = rasterio.merge.merge(mosaic_files)

    # close the connection to the mosaic files
    for file in mosaic_files:
        file.close()

    # crop merged files to the bounds
    with rasterio.io.MemoryFile() as mem_file:
        with mem_file.open(
                driver="GTiff",
                height=merged.shape[1],
                width=merged.shape[2],
                count=merged.shape[0],
                dtype=merged.dtype,
                transform=transform_merged,
        ) as dataset:
            dataset.write(merged)

        with mem_file.open() as dataset:
            cropped, transform_cropped = rasterio.mask.mask(dataset, [bounds], crop=True)

    return cropped, transform_cropped
