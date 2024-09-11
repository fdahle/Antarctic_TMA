"""Finds peaks in a digital elevation model (DEM) array"""

# Library imports
import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import peak_prominences
from tqdm import tqdm


def find_peaks_in_dem(dem: np.ndarray,
                      use_prominence: bool = True,
                      cell_size=10,
                      no_data_value=-99999) -> np.ndarray:
    """
    Finds peaks in a digital elevation model (DEM) array by applying a maximum filter and thresholding.
    The peaks are further refined by applying a relative threshold.

    Args:
        dem (np.ndarray): 2D numpy array representing the DEM.

    Returns:
        TODO: IS NUMPY ARRAY
        list[tuple[int, int]]: List of coordinates of the detected peaks as (col, row) tuples.
    """

    # copy the dem to avoid changing the original
    dem = np.copy(dem)

    # Ignore no_data_value
    valid_mask = (dem != no_data_value)

    # Generate a neighborhood structure
    print("Generate neighborhood structure")
    neighborhood = ndimage.generate_binary_structure(2, 2)

    # Find local maxima
    print("Find local maxima")
    local_max = ndimage.maximum_filter(dem, footprint=neighborhood) == dem

    # Combine the two masks
    peaks = local_max & valid_mask

    # Get the coordinates of the peaks
    peak_coords = np.argwhere(peaks)

    if use_prominence:

        # Convert the 2D coordinates into flat indices
        peak_indices = np.ravel_multi_index((peak_coords[:, 0], peak_coords[:, 1]), dem.shape)

        # Calculate prominences of the peaks
        peak_prominences_vals, left_bases, right_bases = peak_prominences(dem.flatten(), peak_indices)

        # Define number of grid cells
        num_cells_x = cell_size
        num_cells_y = cell_size

        # Calculate grid cell size
        cell_size_x = dem.shape[1] // num_cells_x
        cell_size_y = dem.shape[0] // num_cells_y

        # Initialize a list to store the most prominent peaks in each cell
        most_prominent_peaks = []

        print("Find peaks per cell:")

        # create a progress bar
        pbar = tqdm(total=num_cells_x * num_cells_y)

        # Iterate over the grid
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                pbar.update(1)
                # Define the cell boundaries
                cell_i_min, cell_i_max = i * cell_size_y, min((i + 1) * cell_size_y, dem.shape[0])
                cell_j_min, cell_j_max = j * cell_size_x, min((j + 1) * cell_size_x, dem.shape[1])

                # Find peaks within the cell
                cell_mask = (peak_coords[:, 0] >= cell_i_min) & (peak_coords[:, 0] < cell_i_max) & (
                        peak_coords[:, 1] >= cell_j_min) & (peak_coords[:, 1] < cell_j_max)
                cell_peaks = peak_coords[cell_mask]
                cell_prominences = peak_prominences_vals[cell_mask]

                # Select the most prominent peak in the cell
                if cell_peaks.size > 0:
                    most_prominent_peak = cell_peaks[np.argmax(cell_prominences)]
                    most_prominent_peaks.append(most_prominent_peak)

        # Convert to numpy array for easy plotting
        most_prominent_peaks = np.array(most_prominent_peaks)

        # switch the coordinates from row/col to x/y
        most_prominent_peaks = np.flip(most_prominent_peaks, axis=1)

        print(f"{most_prominent_peaks.shape[0]} peaks found")

        return most_prominent_peaks
    else:

        # Switch the coordinates from row/col to x/y
        peak_coords = np.flip(peak_coords, axis=1)

        print(f"{peak_coords.shape[0]} peaks found")

        return peak_coords


if __name__ == "__main__":
    import numpy as np
    import src.dem.snippets.find_peaks_in_DEM as fpiD

    path_dem = "/home/fdahle/Desktop/agi_test/output/dem_relative.tif"
    no_data_val = -32767

    # load a dem
    import rasterio

    with rasterio.open(path_dem) as src:
        tst_dem = src.read(1)

    # find the peaks of a dem
    tst_peaks = fpiD.find_peaks_in_DEM(tst_dem, no_data_value=no_data_val)

    import src.display.display_images as di

    tst_dem[tst_dem == no_data_val] = np.nan

    di.display_images([tst_dem], points=[tst_peaks])
