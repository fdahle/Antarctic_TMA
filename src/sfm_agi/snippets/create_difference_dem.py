import numpy as np

from scipy.ndimage import convolve


import src.base.resize_image as ri
import src.load.load_rema as lr

def create_difference_dem(historic_dem, modern_dem=None,
                          historic_bounds=None,
                          modern_source="REMA32",
                          dem_nodata=-9999,
                          absolute=False):


    # load the modern dem if not provided
    if modern_dem is None:

        # historic bounds are required to load the modern dem
        if historic_bounds is None:
            raise ValueError("historic_bounds must be provided if modern_dem is not provided")

        if modern_source == "REMA2":
            zoom_level = 2
        elif modern_source == "REMA10":
            zoom_level = 10
        elif modern_source == "REMA32":
            zoom_level = 32
        else:
            raise ValueError("modern source not supported")

        modern_dem, _ = lr.load_rema(historic_bounds, zoom_level=zoom_level)

    # resize the historic dem to the modern dem
    temp_dem = ri.resize_image(historic_dem, modern_dem.shape)

    # calculate the difference
    difference = modern_dem - temp_dem

    # resize back to the original size
    difference = ri.resize_image(difference, historic_dem.shape)

    print(historic_dem.shape, modern_dem.shape, difference.shape)

    # set nodata values to nan
    difference[historic_dem == dem_nodata] = np.nan

    # set pixels to nan if a neighbour is nan
    difference = _set_nan_if_neighbor_is_nan(difference)


    if absolute:
        difference = np.abs(difference)

    return difference

def _set_nan_if_neighbor_is_nan(array):
    # Define a kernel that checks the 8 neighboring pixels
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Create a mask of NaNs
    nan_mask = np.isnan(array).astype(int)

    # Convolve the mask with the kernel
    neighbor_nan_count = convolve(nan_mask, kernel, mode='constant', cval=0)

    # Set pixels to NaN if any neighbor is NaN
    array[neighbor_nan_count > 0] = np.nan

    return array
