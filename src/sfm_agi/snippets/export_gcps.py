import numpy as np
import pandas as pd

import src.dem.snippets.find_peaks_in_DEM as fpiD
import src.dem.snippets.get_elevation_for_points as gefp  # noqa
import src.load.load_rock_mask as lrm

def find_gcps(dem: np.ndarray, transform: np.ndarray,
              bounding_box, zoom_level: int,
              resolution: float, use_rock_mask: bool = False,
              mask_resolution: int = 10, mask_buffer=5) -> pd.DataFrame:
    """
    FInd GCPS in the DEM and export them to a csv file.
    Args:
        dem (np.ndarray): DEM created by the SfM pipeline.
        transform (np.ndarray): Transformation matrix.
        bounding_box (BoundingBox): Bounding box of the dem.
        zoom_level (int): Zoom level of the DEM we want to compare to.
        resolution (float): Resolution of the DEM.
        binary_mask (np.ndarray): Optional binary_mask to apply to the gcps and only
            export gcps that are in 1 in the mask.
        mask_resolution (int): Resolution of the mask in meters.
        mask_buffer (int): Buffer for the mask in pixels
    Returns:

    """

    # identify peaks in the DEM
    px_peaks = fpiD.find_peaks_in_dem(dem, no_data_value=-9999)

    # split the peaks into coordinates and heights
    peak_coords = px_peaks[:, :2]

    # convert px coordinates of peaks to absolute coordinates
    ones = np.ones((peak_coords.shape[0], 1))
    px_peaks = np.hstack([peak_coords, ones])
    a_peaks = px_peaks @ transform.T
    absolute_coords = a_peaks[:, :2]

    # get relative coords (take resolution into account)
    rel_coords = peak_coords * resolution
    rel_coords[:, 0] = bounding_box.min[0] + rel_coords[:, 0]
    rel_coords[:, 1] = bounding_box.max[1] - rel_coords[:, 1]

    # get elevation of the peaks
    elevations_relative = gefp.get_elevation_for_points(peak_coords,
                                                        dem=dem,
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

    if use_rock_mask:

        min_abs_x, min_abs_y = absolute_coords.min(axis=0)
        max_abs_x, max_abs_y = absolute_coords.max(axis=0)

        # get the bounds and load the dem
        absolute_bounds = (min_abs_x, min_abs_y, max_abs_x, max_abs_y)

        print("ABO", absolute_bounds)

        # get the rock mask
        rock_mask = lrm.load_rock_mask(absolute_bounds, mask_resolution)

        # give a warning if no rocks are existing in the mask
        if np.sum(rock_mask) == 0:
            print("WARNING: No rocks found in the rock mask (export_gcps.py)")

        y_mask_coords = (df['y_abs'].values - min_abs_y) / mask_resolution
        x_mask_coords = (df['x_abs'].values - min_abs_x) / mask_resolution

        # cast to int
        y_mask_coords = y_mask_coords.astype(int)
        x_mask_coords = x_mask_coords.astype(int)

        # Check if the coordinates are within the bounds of the DEM
        valid_coords = (y_mask_coords >= 0) & (y_mask_coords < rock_mask.shape[0]) & \
                       (x_mask_coords >= 0) & (x_mask_coords < rock_mask.shape[1])

        # Filter only the valid points
        x_mask_coords = x_mask_coords[valid_coords]
        y_mask_coords = y_mask_coords[valid_coords]
        df = df[valid_coords]

        # create list of tuples for the points
        points = [(x, y) for x, y in zip(x_mask_coords, y_mask_coords)]

        print(y_mask_coords)
        print(x_mask_coords)

        print(np.amin(y_mask_coords), np.amax(y_mask_coords))
        print(np.amin(x_mask_coords), np.amax(x_mask_coords))
        print(rock_mask.shape)

        # apply the rock mask to the gcps
        mask = rock_mask[y_mask_coords, x_mask_coords] == 1

        import src.display.display_images as di
        di.display_images([rock_mask],
                          points=[points])

        print(mask)

        # Return the filtered dataframe
        print(df.shape)
        df = df[mask]
        print(df.shape)

        exit()

    return df
