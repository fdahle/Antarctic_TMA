"""Finds peaks in a digital elevation model (DEM) array"""

# Library imports
import numpy as np
import scipy.ndimage as ndimage


def find_peaks(dem: np.ndarray, threshold_abs: float = 0, threshold_rel: float = 0.5) -> list[tuple[int, int]]:
    """
    Finds peaks in a digital elevation model (DEM) array by applying a maximum filter and thresholding.
    The peaks are further refined by applying a relative threshold.

    Args:
        dem (np.ndarray): 2D numpy array representing the DEM.
        threshold_abs (float, optional): Absolute threshold for peak detection. Peaks below this value will be ignored.
            Defaults to 0.
        threshold_rel (float, optional): Relative threshold for refining peak detection. Defaults to 0.5.

    Returns:
        list[tuple[int, int]]: List of coordinates of the detected peaks as (col, row) tuples.
    """
    neighborhood = ndimage.generate_binary_structure(2, 2)
    local_max = ndimage.maximum_filter(dem, footprint=neighborhood) == dem

    # Apply the absolute threshold
    detected_peaks = (dem > threshold_abs)

    # Combine the two masks
    peaks = local_max & detected_peaks

    # Get the coordinates of the peaks
    peak_coords = np.argwhere(peaks)

    # Further refine peaks by relative threshold
    refined_peaks = []
    for (row, col) in peak_coords:
        window = dem[max(0, row - 1):row + 2, max(0, col - 1):col + 2]
        if dem[row, col] >= np.max(window) * threshold_rel:
            refined_peaks.append((col, row))

    return refined_peaks
