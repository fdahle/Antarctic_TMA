import os.path
import numpy as np
import rasterio
from scipy.spatial import cKDTree


def create_confidence_arr(dem, point_cloud, transform,
                          interpolate=False, distance=10,
                          dem_nodata=-9999, min_confidence=0):

    # convert no data value to nan
    dem[dem == dem_nodata] = np.nan

    # convert the transform to a numpy array
    if type(transform) is rasterio.Affine:
        transform = np.asarray(transform).reshape(3, 3)

    # Apply the inverse of the affine transform to map x, y to pixel coordinates
    inv_transform = np.linalg.inv(transform)

    # Extract x, y, and confidence values
    x_coords = point_cloud[:, 0]
    y_coords = point_cloud[:, 1]
    confidences = point_cloud[:, 3]

    # Convert point cloud world coordinates to DEM pixel coordinates
    ones = np.ones_like(x_coords)
    points_homogeneous = np.vstack((x_coords, y_coords, ones))
    pc_coords = inv_transform @ points_homogeneous
    pc_coords = pc_coords[:2].astype(int)  # Get row, col coordinates

    # Clip coordinates to ensure they fall within the DEM bounds
    row_indices = np.clip(pc_coords[1], 0, dem.shape[0] - 1)
    col_indices = np.clip(pc_coords[0], 0, dem.shape[1] - 1)

    # Create arrays to hold the sum of confidences and count of points for each DEM cell
    confidence_sum = np.zeros(dem.shape)
    count = np.zeros(dem.shape)

    # Accumulate confidence values and count occurrences for each DEM cell
    for row, col, conf in zip(row_indices, col_indices, confidences):
        confidence_sum[row, col] += conf
        count[row, col] += 1

    # Calculate the average confidence for each DEM cell
    with np.errstate(divide='ignore', invalid='ignore'):
        confidence = np.true_divide(confidence_sum, count)

    if interpolate:

        # Get the coordinates of the pixels with known confidence values
        known_indices = np.where(~np.isnan(confidence))
        known_points = np.vstack(known_indices).T
        known_values = confidence[known_indices]

        # Use cKDTree to find nearest neighbors within the max_distance
        tree = cKDTree(known_points)
        grid_x, grid_y = np.meshgrid(np.arange(dem.shape[1]), np.arange(dem.shape[0]))
        grid_points = np.vstack([grid_y.ravel(), grid_x.ravel()]).T

        # Find the nearest neighbors and distances
        distances, indices = tree.query(grid_points, distance_upper_bound=distance)
        confidence = np.full(dem.shape, np.nan)

        # Interpolate the confidence values
        valid_mask = distances < distance
        confidence[grid_y.ravel()[valid_mask], grid_x.ravel()[valid_mask]] = known_values[indices[valid_mask]]

    # Handle cells with no points
    confidence[np.isnan(confidence)] = min_confidence
    confidence[np.isnan(dem)] = np.nan

    return confidence
