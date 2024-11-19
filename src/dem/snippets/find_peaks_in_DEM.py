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

    # get the nanmin and nanmax of the dem
    dem_min = np.nanmin(dem)
    dem_max = np.nanmax(dem)

    # adapt minimum threshold value based on dem_min and dem_max
    range_factor = 0.05
    peak_threshold = (dem_max - dem_min) * range_factor

    # define min distance between peaks
    min_distance = int(cell_size * 0.1)

    print("peak threshold: ", peak_threshold)
    print("min distance: ", min_distance)

    # Apply Gaussian filter to smooth DEM (optional, helps reduce noise in peak detection)
    if smoothing:
        _dem = gaussian_filter(dem, sigma=1)

    # check how many cells fit in the dem
    num_cells_x = dem.shape[1] // cell_size
    num_cells_y = dem.shape[0] // cell_size

    # Initialize a list to store the selected peaks in each cell
    selected_peaks = []

    # create a progress bar
    pbar = tqdm(total=num_cells_x * num_cells_y)

    # Iterate over the grid
    for i in range(num_cells_y):
        for j in range(num_cells_x):

            pbar.update(1)
            pbar.set_description(f"Find peaks")
            pbar.set_postfix_str(f"Cell at {i+1}, {j+1}")

            # Define the cell boundaries
            cell_min_x = j * cell_size
            cell_max_x = (j + 1) * cell_size
            cell_min_y = i * cell_size
            cell_max_y = (i + 1) * cell_size

            # get the cell
            cell = dem[cell_min_y:cell_max_y, cell_min_x:cell_max_x]

            # skip cells with less than X percent of data
            if np.isnan(cell).sum() > 0.25 * cell_size * cell_size:
                continue

            # find all peaks in the cell
            peak_coords = peak_local_max(cell, min_distance=min_distance,
                                         threshold_abs=peak_threshold)

            # switch x and y
            peak_coords = np.flip(peak_coords, axis=1)

            # get the height of the peaks
            cell_heights = cell[peak_coords[:, 1], peak_coords[:, 0]]

            # get the most high peak of each cell
            highest_peak = cell[np.argmax(cell_heights)]

            print(highest_peak)
            exit()

            # append the most prominent peak to the numpy array
            selected_peaks.append(highest_peak)

    pbar.set_postfix_str("Finished!")

    # convert to numpy array
    selected_peaks = np.array(selected_peaks)

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
