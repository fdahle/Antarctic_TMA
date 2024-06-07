import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal

def find_peaks(dem, threshold_abs=0, threshold_rel=0.5):

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