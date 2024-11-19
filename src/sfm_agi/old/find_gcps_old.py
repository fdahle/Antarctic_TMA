import copy

import numpy as np
import pandas as pd

import src.dem.snippets.get_elevation_for_points as gefp  # noqa
import src.display.display_images as di
import src.base.resize_image as ri

# debug variables
debug_show_points = True


def find_gcps(hist_dem: np.ndarray,
              modern_dem: np.ndarray,
              transform: np.ndarray,
              bounding_box, zoom_level: int,
              resolution: float,
              mask: np.ndarray | None = None,
              ortho: np.ndarray | None = None,
              no_data_value: int = -9999,
              ) -> pd.DataFrame:
    """
    Find GCPS in the DEM and export them to a csv file.
    Args:
        hist_dem (np.ndarray): DEM created by the SfM pipeline.
        transform (np.ndarray): Transformation matrix.
        bounding_box (BoundingBox): Bounding box of the dem.
        zoom_level (int): Zoom level of the DEM we want to compare to.
        resolution (float): Resolution of the DEM.
    Returns:

    """

    modern_dem = ri.resize_image(modern_dem, hist_dem.shape)
    modern_dem[np.isnan(hist_dem)] = np.nan

    # set the dems to none where the mask is 0
    if mask is not None:
        hist_dem[mask == 0] = np.nan
        modern_dem[mask == 0] = np.nan

    range_factor = 0.05

    # temp copy of hist dem with nan values for no data
    hist_dem_temp = copy.deepcopy(hist_dem)
    hist_dem_temp[hist_dem_temp == no_data_value] = np.nan

    # calculate peak_threshold for hist and modern dem
    hist_peak_threshold = (np.nanmax(hist_dem_temp) - np.nanmin(hist_dem_temp)) * range_factor
    modern_peak_threshold = (np.nanmax(modern_dem) - np.nanmin(modern_dem)) * range_factor

    print("max min hist", np.nanmax(hist_dem_temp), np.nanmin(hist_dem_temp))
    print("max min modern", np.nanmax(modern_dem), np.nanmin(modern_dem))

    print("hist_peak_threshold", hist_peak_threshold)
    print("modern_peak_threshold", modern_peak_threshold)

    cell_size = 250
    min_distance = int(cell_size * 0.1)

    # check how many cells fit in the dem
    num_cells_x = hist_dem.shape[1] // cell_size
    num_cells_y = hist_dem.shape[0] // cell_size

    from skimage.feature import peak_local_max

    hist_px_peaks = np.empty((0, 2))
    modern_px_peaks = np.empty((0, 2))

    # Iterate over the grid
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            # Define the cell boundaries
            cell_min_x = j * cell_size
            cell_max_x = (j + 1) * cell_size
            cell_min_y = i * cell_size
            cell_max_y = (i + 1) * cell_size

            # get the cell
            hist_cell = hist_dem[cell_min_y:cell_max_y, cell_min_x:cell_max_x]
            modern_cell = modern_dem[cell_min_y:cell_max_y, cell_min_x:cell_max_x]

            print(hist_cell.shape, modern_cell.shape)

            hist_cell = copy.deepcopy(hist_cell)
            hist_cell[hist_cell == no_data_value] = np.nan
            hist_cell[np.isnan(hist_cell)] = np.nanmin(hist_cell)
            hist_cell -= np.nanmin(hist_cell)

            # skip cells with less than X percent of data
            #if np.isnan(hist_cell).sum() > 0.25 * cell_size * cell_size:
            #    continue
            #if np.isnan(modern_cell).sum() > 0.25 * cell_size * cell_size:
            #    continue

            # find all peaks in the cell
            #hist_peak_coords = peak_local_max(hist_cell, min_distance=min_distance,
            #                             threshold_abs=hist_peak_threshold)
            #modern_peak_coords = peak_local_max(modern_cell, min_distance=min_distance,
            #                                threshold_abs=modern_peak_threshold)

            # skip if all is nan
            if np.isnan(hist_cell).all() or np.isnan(modern_cell).all():
                continue

            # Find the index of the maximum elevation
            hist_max_index = np.argmax(hist_cell)
            hist_peak_coords = np.array(np.unravel_index(hist_max_index,
                                                         hist_cell.shape))
            hist_peak_coords = hist_peak_coords.reshape(1, 2)
            modern_max_index = np.argmax(modern_cell)
            modern_peak_coords = np.array(np.unravel_index(modern_max_index,
                                                           modern_cell.shape))
            modern_peak_coords = modern_peak_coords.reshape(1, 2)
            print("hist_peak_coords", hist_peak_coords)
            print("modern_peak_coords", modern_peak_coords)

            # add the cell offset to the peak coordinates
            hist_peak_coords[:, 0] += cell_min_x
            hist_peak_coords[:, 1] += cell_min_y
            modern_peak_coords[:, 0] += cell_min_x
            modern_peak_coords[:, 1] += cell_min_y

            # save the peaks
            hist_px_peaks = np.vstack([hist_px_peaks, hist_peak_coords])
            modern_px_peaks = np.vstack([modern_px_peaks, modern_peak_coords])


            if hist_peak_coords.shape[0] == 0 and modern_peak_coords.shape[0] == 0:
                continue

            di.display_images([hist_cell, modern_cell],
                                image_types=['dem', 'dem'],
                              points=[hist_peak_coords, modern_peak_coords])

    print(hist_px_peaks.shape)
    print(modern_px_peaks.shape)
    hist_dem[hist_dem == no_data_value] = np.nan
    hist_dem[hist_dem == np.nan] = np.nanmin(hist_dem)

    # scale modern dem to min and max of hist dem
    mean_historical = np.nanmean(hist_dem)
    std_historical = np.nanstd(hist_dem)

    mean_modern = np.nanmean(modern_dem)
    std_modern = np.nanstd(modern_dem)

    # Scale the modern DEM to align with the historical DEM
    scaled_modern_dem = (std_historical / std_modern) * (modern_dem - mean_modern) + mean_historical

    di.display_images([hist_dem, scaled_modern_dem], image_types=['dem', 'dem'],
                      points=[hist_px_peaks, modern_px_peaks])

    # split the peaks into coordinates and heights
    peak_coords = hist_px_peaks[:, :2]

    # convert px coordinates of peaks to absolute coordinates
    ones = np.ones((peak_coords.shape[0], 1))
    hist_px_peaks = np.hstack([peak_coords, ones])
    a_peaks = hist_px_peaks @ transform.T
    absolute_coords = a_peaks[:, :2]

    # get relative coords (take resolution into account)
    rel_coords = peak_coords * resolution
    rel_coords[:, 0] = bounding_box.min[0] + rel_coords[:, 0]
    rel_coords[:, 1] = bounding_box.max[1] - rel_coords[:, 1]

    # get elevation of the peaks
    elevations_relative = gefp.get_elevation_for_points(peak_coords,
                                                        dem=hist_dem,
                                                        point_type="relative")
    elevations_relative = elevations_relative.reshape(-1, 1)
    elevations_absolute = gefp.get_elevation_for_points(absolute_coords,
                                                        zoom_level=zoom_level,
                                                        point_type="absolute")
    elevations_absolute = elevations_absolute.reshape(-1, 1)

    # concat all data
    peaks = np.hstack([peak_coords,
                       rel_coords, elevations_relative,
                       absolute_coords, elevations_absolute])

    # Create a DataFrame
    df = pd.DataFrame(peaks, columns=['x_px', 'y_px',
                                      'x_rel', 'y_rel', 'z_rel',
                                      'x_abs', 'y_abs', 'z_abs'])

    if mask is not None:

        # Convert the x_px and y_px columns to integers to index the mask
        x_px = df['x_px'].astype(int)
        y_px = df['y_px'].astype(int)

        # Ensure coordinates are within bounds
        x_px = np.clip(x_px, 0, mask.shape[1] - 1)
        y_px = np.clip(y_px, 0, mask.shape[0] - 1)

        # Filter points where the mask is 1
        mask_values = mask[y_px, x_px]  # This gives the mask values at the (x_px, y_px) coordinates
        df_ignored = df[mask_values == 0]
        df = df[mask_values == 1]

        # reset the index
        df_ignored.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        print("Number of GCPs ignored due to mask:", df_ignored.shape[0])
        print(df.shape[0], "GCPs are kept")

    # show the gcps
    if debug_show_points:



        # replace mask with a default mask for display
        if mask is None:
            mask = np.ones_like(hist_dem)
            df_ignored = pd.DataFrame(columns=df.columns)

        points = np.vstack([df['x_px'].values, df['y_px'].values]).T
        points_ignored = np.vstack([df_ignored['x_px'].values, df_ignored['y_px'].values]).T
        style_config = {
            "titles_x": ["Points", "ignored Points"]
        }
        if ortho is None:
            di.display_images([hist_dem, hist_dem, mask, mask],
                              image_types=['dem', 'dem', 'binary', 'binary'],
                              points=[points, points_ignored, points, points_ignored],
                              style_config=style_config)
        else:
            style_config["plot_shape"] = (3, 2)
            di.display_images([hist_dem, hist_dem, mask, mask, ortho, ortho],
                              image_types=['dem', 'dem', 'binary', 'binary', 'gray', 'gray'],
                              points=[points, points_ignored, points, points_ignored, points, points_ignored],
                              style_config=style_config)

    return df
