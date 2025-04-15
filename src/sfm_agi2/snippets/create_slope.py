""" Compute the slope in degrees from a Digital Elevation Model (DEM). """

import numpy as np
import xdem

from affine import Affine
from scipy.ndimage import convolve


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
    #dem = xdem.DEM.from_array(dem, transform, epsg_code, nodata=no_data)

    # get slope
    #slope = xdem.terrain.slope(dem).data

    # Get pixel resolution
    pixel_size = transform.a

    # Prepare mask for no-data
    if np.isnan(no_data):
        valid_mask = ~np.isnan(dem)
    else:
        valid_mask = dem != no_data
        dem[~valid_mask] = np.nan

    # Horn's method convolution kernels
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]) / (8 * pixel_size)

    kernel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]) / (8 * pixel_size)

    # NaN-safe convolution
    dz_dx = convolve(np.nan_to_num(dem, nan=0.0), kernel_x, mode='nearest')
    dz_dy = convolve(np.nan_to_num(dem, nan=0.0), kernel_y, mode='nearest')

    # Convolve while ignoring NaNs
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_deg = np.degrees(slope).astype(np.float32)

    slope_deg[~valid_mask] = np.nan  # Apply no-data mask
    return slope_deg