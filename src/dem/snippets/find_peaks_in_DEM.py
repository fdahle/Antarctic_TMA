import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
from skimage.feature import peak_local_max
from tqdm import tqdm

import src.display.display_images as di

debug_display_peaks = False


def find_peaks_in_dem(dem,
                      smoothing=True,
                      cell_size=250,
                      no_data_value=-9999):

    # copy the dem to avoid changing the original
    _dem = np.copy(dem)

    # set no data to nan
    _dem[_dem == no_data_value] = np.nan

    # Find indices where rows/columns contain only NaNs
    nan_rows = np.isnan(_dem).all(axis=1)
    nan_cols = np.isnan(_dem).all(axis=0)

    # Calculate counts of NaN-only rows and columns at the top and left
    rows_lost_top = np.argmax(~nan_rows)
    cols_lost_left = np.argmax(~nan_cols)

    # remove lines and columns with only NaN values
    _dem = _dem[~np.isnan(_dem).all(axis=1)]
    _dem = _dem[:, ~np.isnan(_dem).all(axis=0)]

    # standardize dem so that the minimum value is 0
    _dem -= np.nanmin(_dem)

    # Apply Gaussian filter to smooth DEM (optional, helps reduce noise in peak detection)
    if smoothing:
        _dem = gaussian_filter(_dem, sigma=1)

    # Find peaks in the DEM
    peaks_coords = peak_local_max(_dem, min_distance=10, threshold_abs=0, exclude_border=False)

    # Extract the x, y coordinates of the peaks
    y_peaks, x_peaks = peaks_coords[:, 0], peaks_coords[:, 1]

    # merge the x and y coordinates
    peak_coords = np.asarray(list(zip(x_peaks, y_peaks)))

    # check how many cells fit in the dem
    num_cells_x = _dem.shape[1] // cell_size
    num_cells_y = _dem.shape[0] // cell_size

    # Initialize a list to store the selected peaks in each cell
    selected_peaks = []

    # create a progress bar
    pbar = tqdm(total=num_cells_x * num_cells_y)

    # Iterate over the grid
    for i in range(num_cells_y):
        for j in range(num_cells_x):

            pbar.update(1)
            pbar.set_description(f"Find most prominent peaks")
            pbar.set_postfix_str(f"Cell at {i+1}, {j+1}")

            # Define the cell boundaries
            cell_min_x = j * cell_size
            cell_max_x = (j + 1) * cell_size
            cell_min_y = i * cell_size
            cell_max_y = (i + 1) * cell_size

            # get all peaks within the cell
            cell_indices = (peak_coords[:, 0] >= cell_min_x) & (peak_coords[:, 0] < cell_max_x) & (
                    peak_coords[:, 1] >= cell_min_y) & (peak_coords[:, 1] < cell_max_y)
            cell_peaks = peak_coords[cell_indices]

            # skip cells with no peaks
            if cell_peaks.shape[0] == 0:
                continue

            # get the height of the peaks
            cell_heights = _dem[cell_peaks[:, 1], cell_peaks[:, 0]]

            # get the most high peak of each cell
            highest_peak = cell_peaks[np.argmax(cell_heights)]

            # append the most prominent peak to the numpy array
            selected_peaks.append(highest_peak)

    pbar.set_postfix_str("Finished!")

    # convert to numpy array
    selected_peaks = np.array(selected_peaks)

    # add the lost rows and columns back to the peak coordinates
    peak_coords[:, 0] += cols_lost_left
    peak_coords[:, 1] += rows_lost_top
    selected_peaks[:, 0] += cols_lost_left
    selected_peaks[:, 1] += rows_lost_top

    if debug_display_peaks:

        di.display_images([dem, dem],
                          points=[peak_coords, selected_peaks])

    return selected_peaks

if __name__ == "__main__":

    project_name = "gcp_test2"
    dem_path = f"/data/ATM/data_1/sfm/agi_projects/{project_name}/output/{project_name}_dem_relative.tif"

    import src.load.load_image as li
    dem = li.load_image(dem_path, image_type="dem")

    find_peaks_in_dem(dem, smoothing=True)
