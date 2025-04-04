""" Compute the slope in degrees from a Digital Elevation Model (DEM). """

import numpy as np
import xdem

from affine import Affine


def create_slope(dem: np.ndarray,
                 transform: Affine,
                 epsg_code: int,
                 no_data: int | float = np.nan):
    """
    This function uses the `xdem` library to calculate terrain slope from
    a DEM provided as a NumPy array. The slope is returned as a NumPy array
    with the same shape, where each pixel represents the slope in degrees.

    Args:
        dem (np.ndarray): Input DEM as a 2D NumPy array.
        transform (Affine): Affine transformation defining the geo-referencing
            of the DEM.
        epsg_code (int): EPSG code for the coordinate reference system of the DEM.
        no_data (int | float, optional): Value representing no-data areas in the DEM.
            Defaults to np.nan.

    Returns:
        np.ndarray: Slope map in degrees as a 2D NumPy array.
    """

    # convert dem to xdem
    dem = xdem.DEM.from_array(dem, transform, epsg_code, nodata=no_data)

    # get slope
    slope = xdem.terrain.slope(dem)

    return slope.data