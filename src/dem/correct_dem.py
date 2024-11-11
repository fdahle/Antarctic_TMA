from tkinter import image_types

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import match_template
import src.display.display_images as di


def correct_dem(historic_dem, modern_dem, mode='nuth', remove_outliers=True, max_offset=100,
                max_dz=100, slope_lim=(0.1, 40), max_iter=30, tol=0.02, plot=False, mask=None):
    """
    Perform DEM co-registration using numpy arrays for historic and modern DEMs.

    Parameters:
    - historic_dem (numpy array): Reference DEM.
    - modern_dem (numpy array): Source DEM to be aligned.
    - mode (str): Mode of co-registration, choices: 'ncc', 'sad', 'nuth', 'none'.
    - remove_outliers (bool): Whether to apply outlier filtering.
    - max_offset (float): Maximum expected horizontal offset in pixels.
    - max_dz (float): Maximum allowed elevation difference for outlier filtering.
    - slope_lim (tuple): Slope limits for filtering surfaces (min, max).
    - max_iter (int): Maximum number of iterations for refinement.
    - tol (float): Tolerance to stop iteration when shift magnitude falls below.
    - plot (bool): Whether to plot final results (requires matplotlib).

    Returns:
    - dx_total, dy_total, dz_total (floats): Total computed shifts in x, y, and z.
    - final_diff (numpy array): Final difference map after alignment.
    """

    # Apply mask to exclude invalid areas if provided
    if mask is not None:
        historic_dem = np.where(mask, historic_dem, np.nan)
        modern_dem = np.where(mask, modern_dem, np.nan)

    # Initialize shifts
    dx_total, dy_total, dz_total = 0, 0, 0

    # Iteratively compute shifts until tolerance or max_iter is reached
    for n in range(max_iter):
        diff = modern_dem - historic_dem

        if remove_outliers:
            diff = _outlier_filter(diff, max_dz=max_dz)

        # Check if diff is all NaNs; if so, skip this iteration
        if np.isnan(diff).all():
            print("All values in diff are NaN, skipping iteration.")
            continue

        # Slope filtering
        slope = _get_filtered_slope(historic_dem, slope_lim=slope_lim)

        # Check if slope is all NaNs; if so, skip this iteration
        if np.isnan(slope).all():
            print("All values in slope are NaN, skipping iteration.")
            continue

        aspect = np.arctan2(*np.gradient(historic_dem))

        # Compute shift based on the selected mode
        dx, dy = 0, 0
        if mode == 'ncc':
            result = match_template(historic_dem, modern_dem, pad_input=True)
            y_offset, x_offset = np.unravel_index(np.argmax(result), result.shape)
            dy, dx = y_offset - historic_dem.shape[0] // 2, x_offset - historic_dem.shape[1] // 2
        elif mode == 'nuth':
            dx, dy = _compute_nuth_shift(diff, slope, aspect)

        # If dx or dy is NaN, set them to zero to avoid errors in np.roll
        dx = 0 if np.isnan(dx) else dx
        dy = 0 if np.isnan(dy) else dy

        # Apply shifts
        dx_total += dx
        dy_total += dy
        dz_total += np.nanmedian(diff) if not np.isnan(np.nanmedian(diff)) else 0

        print(f"Iteration {n}: dx={dx}, dy={dy}, dz_total={dz_total}")

        # Update modern_dem by shifting in x, y, and z
        shifted_x = np.roll(historic_dem, int(round(dx)), axis=1)  # Ensure dx is an integer
        shifted_xy = np.roll(shifted_x, int(round(dy)), axis=0)  # Ensure dy is an integer
        historic_dem = shifted_xy + dz_total  # Update historic_dem with cumulative z-shift

        style_config = {
            "title": f"{np.nanmean(diff):.2f} m",
        }
        #di.display_images([historic_dem, modern_dem, diff], image_types=["DEM", "DEM", "difference"],
                          #style_config=style_config)

        # Check for convergence
        print(np.sqrt(dx**2 + dy**2 + dz_total**2), tol)
        if np.sqrt(dx**2 + dy**2 + dz_total**2) < tol:
            break

    # Compute final difference after alignment
    final_diff = modern_dem - historic_dem

    # Optional plot of final alignment
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(historic_dem, cmap='gray')
        plt.title("Aligned Historic DEM")
        plt.subplot(1, 2, 2)
        plt.imshow(final_diff, cmap='RdBu', vmin=-max_dz, vmax=max_dz)
        plt.title("Final Elevation Difference (After Alignment)")
        plt.colorbar(label="Elevation Difference (m)")
        plt.show()

    return dx_total, dy_total, dz_total, final_diff


def _outlier_filter(diff, max_dz=100):
    diff = np.where(np.abs(diff) > max_dz, np.nan, diff)
    mad = np.nanmedian(np.abs(diff - np.nanmedian(diff)))
    threshold = 3 * mad
    return np.where(np.abs(diff - np.nanmedian(diff)) > threshold, np.nan, diff)

def _get_filtered_slope(dem, slope_lim=(0.1, 40)):
    gradient_x, gradient_y = np.gradient(dem)
    slope = np.sqrt(gradient_x**2 + gradient_y**2)
    return np.where((slope >= slope_lim[0]) & (slope <= slope_lim[1]), slope, np.nan)

def _compute_nuth_shift(diff, slope, aspect):
    fit_param = np.nanmedian(diff) / np.tan(np.nanmedian(slope))
    dx = fit_param * np.sin(aspect)
    dy = fit_param * np.cos(aspect)
    # Calculate the mean dx and dy to get a single global shift
    return np.nanmean(dx), np.nanmean(dy)  # Now returns scalar values


if __name__ == "__main__":

    project_fld = "/data/ATM/data_1/sfm/agi_projects/"
    project_name = "new_tst8"

    import os
    project_path = os.path.join(project_fld, project_name)
    dem_path = os.path.join(project_path, "output", f"{project_name}_dem_absolute.tif")

    import src.load.load_image as li
    import src.load.load_rema as lr

    # load the historic DEM
    hist_dem, hist_transform = li.load_image(dem_path, return_transform=True)

    # get bounds from the transform and the shape of the image
    import src.base.calc_bounds as cb
    bounds = cb.calc_bounds(hist_transform, hist_dem.shape)

    modern_dem, modern_transform = lr.load_rema(bounds)

    # resize the modern DEM to the same size as the historic DEM
    import src.base.resize_image as ri
    modern_dem = ri.resize_image(modern_dem, hist_dem.shape)

    hist_dem[hist_dem == hist_dem.min()] = np.nan

    print("TEMP")
    hist_dem[hist_dem > 1000] = np.nan

    mask = np.invert(np.isnan(hist_dem) | np.isnan(modern_dem))

    _, _, _, corrected_dem = correct_dem(hist_dem, modern_dem, mask=mask)
    diff_hist = np.abs(hist_dem - modern_dem)
    diff_corrected = np.abs(corrected_dem - modern_dem)
    placeholder = np.zeros_like(diff_hist)

    di.display_images([hist_dem, corrected_dem, modern_dem,
                        diff_hist, diff_corrected, placeholder],
                      image_types=["DEM", "DEM", "DEM",
                                   "difference", "difference", "difference"],)
