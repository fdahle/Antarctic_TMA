"""" create a difference array between two DEMs"""

# Library imports
import numpy as np
from scipy.ndimage import convolve

# Local imports
import src.base.resize_image as ri
import src.load.load_rema as lr

def create_difference_dem(historic_dem: np.ndarray,
                          modern_dem: np.ndarray | None = None,
                          historic_bounds: tuple[float, float, float, float] | None = None,
                          modern_source: str = "REMA32",
                          dem_nodata: float = -9999,
                          absolute: bool = False):
    """
        Create a difference DEM by subtracting a historic DEM from a modern DEM.
        Args:
            historic_dem (np.ndarray): The historic DEM as a NumPy array.
            modern_dem (Optional[np.ndarray]): The modern DEM as a NumPy array.
                If None, the modern DEM will be loaded using the historic bounds.
            historic_bounds (Optional[Tuple[float, float, float, float]]): The
                bounding box of the historic DEM (min_x, min_y, max_x, max_y). Required
                if `modern_dem` is not provided.
            modern_source (str): The source of the modern DEM
                ("REMA2", "REMA10", "REMA32"). Defaults to "REMA32".
            dem_nodata (float): The value used to indicate no data in the DEM.
                Defaults to -9999.
            absolute (bool): If True, returns the absolute difference.
                Defaults to False.
        Returns:
            np.ndarray: The difference DEM with the same shape as
                the historic DEM.
        Raises:
            ValueError: If `historic_bounds` is not provided when `modern_dem`
                is None.
            ValueError: If an unsupported `modern_source` is provided.
        """

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

    # get the shape of the historic dem
    historic_shape = (historic_dem.shape[0], historic_dem.shape[1])

    # resize back to the original size
    difference = ri.resize_image(difference, historic_shape)

    # set nodata values to nan
    difference[historic_dem == dem_nodata] = np.nan

    # set pixels to nan if a neighbour is nan
    difference = _set_nan_if_neighbor_is_nan(difference)

    if absolute:
        difference = np.abs(difference)

    # there's a problem with abnormal high values that are not true
    difference[difference > 5000] = np.nan

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
