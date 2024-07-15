import numpy as np
import os
import pandas as pd

import src.dem.find_peaks_in_DEM as fpid
import src.dem.get_elevation_for_points as gefp
import src.sfm_agi.snippets.georef_ortho as go


def export_gcps(dem, ortho, bounding_box, zoom_level,
                resolution, footprints, output_path):

    # get first band of ortho if it has more than one band
    if len(ortho.shape) == 3:
        ortho = ortho[0]

    if dem.shape != ortho.shape:
        raise ValueError("DEM and ortho should have the same shape")

    # convert pandas series to list
    lst_footprints = footprints.to_list()

    # get transform of ortho
    transform = go.georef_ortho(ortho, lst_footprints)

    # identify peaks in the DEM
    px_peaks = fpid.find_peaks_in_dem(dem, no_data_value=-9999)

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

    # Create labels (n x 1 array)
    num_points = absolute_coords.shape[0]
    labels = pd.Series([f"gcp_{i + 1}" for i in range(num_points)])

    # Create a DataFrame
    df = pd.DataFrame(peaks, columns=['x_px', 'y_px',
                                      'x_rel', 'y_rel', 'z_rel',
                                      'x_abs', 'y_abs', 'z_abs'])
    df.insert(0, 'GCP', labels)

    df.to_csv(output_path, sep=';', index=False, float_format='%.8f')
