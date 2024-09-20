import numpy as np
import pandas as pd

import src.dem.snippets.find_peaks_in_DEM as fpiD
import src.dem.snippets.get_elevation_for_points as gefp  # noqa
import src.display.display_images as di


# debug variables
debug_show_points = False


def find_gcps(dem: np.ndarray, transform: np.ndarray,
              bounding_box, zoom_level: int,
              resolution: float, mask: np.ndarray | None = None,
              ) -> pd.DataFrame:
    """
    FInd GCPS in the DEM and export them to a csv file.
    Args:
        dem (np.ndarray): DEM created by the SfM pipeline.
        transform (np.ndarray): Transformation matrix.
        bounding_box (BoundingBox): Bounding box of the dem.
        zoom_level (int): Zoom level of the DEM we want to compare to.
        resolution (float): Resolution of the DEM.
        use_rock_mask (bool): Optional use of rock mask to apply to the gcps
            and only export gcps that are in 1 (=rocks) in the mask.
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
            mask = np.ones_like(dem)
            df_ignored = pd.DataFrame(columns=df.columns)

        points = np.vstack([df['x_px'].values, df['y_px'].values]).T
        points_ignored = np.vstack([df_ignored['x_px'].values, df_ignored['y_px'].values]).T
        di.display_images([dem, dem, mask, mask],
                          image_types=['dem', 'dem', 'binary','binary'],
                          points=[points, points_ignored, points, points_ignored])

    return df
