import numpy as np
import open3d as o3d
import rasterio
from scipy import ndimage

from scipy.interpolate import griddata
from scipy.spatial import cKDTree

def create_confidence_arr(dem: np.ndarray,
                          point_cloud: np.ndarray,
                          transform,
                          interpolate: bool = False, distance: int = 25,
                          dem_nodata: int = -9999, min_confidence=0):

    # check if the point cloud is from open3d
    if isinstance(point_cloud, o3d.geometry.PointCloud):
        raise ValueError("Input point cloud must be a numpy array"
                         " in order to access custom attributes.")

    # check if transform is a numpy array
    if isinstance(transform, np.ndarray):
        # remove last row if it is [0, 0, 1]
        if transform.shape[0] == 9 and np.allclose(transform[-3:], [0, 0, 1]):
            transform = transform[:-3]

        # convert np-array to rasterio transform
        transform = rasterio.transform.Affine(*transform)

    # create the confidence array
    confidence_array = np.zeros_like(dem, dtype=np.float32)

    # Extract X, Y, and confidence values from the point cloud
    x_coords = point_cloud[:, 0]
    y_coords = point_cloud[:, 1]
    confidence_values = point_cloud[:, 10]
    count_array = np.zeros_like(dem, dtype=np.int32)

    # Invert the transform to map world coordinates to pixel coordinates
    inverse_transform = ~transform

    # Convert world coordinates to pixel indices (row, col)
    cols, rows = inverse_transform * (x_coords, y_coords)
    cols = cols.astype(int)
    rows = rows.astype(int)

    # Create a mask to filter out points that fall outside the DEM bounds
    valid_mask = (rows >= 0) & (rows < dem.shape[0]) & (cols >= 0) & (cols < dem.shape[1])
    valid_rows = rows[valid_mask]
    valid_cols = cols[valid_mask]
    valid_confidences = confidence_values[valid_mask]

    # Use numpy indexing to accumulate confidence values and counts
    np.add.at(confidence_array, (valid_rows, valid_cols), valid_confidences)
    np.add.at(count_array, (valid_rows, valid_cols), 1)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        confidence_array = np.divide(confidence_array, count_array, where=(count_array > 0))

    if interpolate:
        print("Interpolating confidence values...")

        # Create a mask of cells with no points.
        missing_mask = (count_array == 0)

        # Compute the distance transform.
        # 'indices' will contain, for each cell, the indices of the nearest non-missing cell.
        distances, indices = ndimage.distance_transform_edt(
            missing_mask, return_distances=True, return_indices=True)

        # Extract the nearest valid cell indices.
        nearest_row = indices[0]
        nearest_col = indices[1]

        # If a maximum distance threshold is desired, create a mask for cells within that threshold.
        threshold_mask = distances <= distance

        # Only update cells that are both missing and within the allowed distance.
        fill_mask = missing_mask & threshold_mask

        # Use the indices from the distance transform to assign the nearest valid confidence.
        confidence_array[fill_mask] = confidence_array[nearest_row[fill_mask], nearest_col[fill_mask]]

        """
        # Create mask for valid values
        valid_mask = count_array > 0
        invalid_mask = ~valid_mask

        # Get coordinates of valid and invalid points
        valid_coords = np.argwhere(valid_mask)
        invalid_coords = np.argwhere(invalid_mask)

        # Values at valid coordinates
        valid_values = confidence_array[valid_mask]

        # Create a KDTree for efficient distance calculation
        print("Build KDTree...")
        tree = cKDTree(valid_coords)

        print("Querying KDTree...")
        # Query invalid points within distance threshold
        distances, indices = tree.query(invalid_coords, k=1, distance_upper_bound=distance)

        # Mask for valid results
        nearby_mask = distances != np.inf
        nearby_invalid_coords = invalid_coords[nearby_mask]
        nearby_indices = indices[nearby_mask]

        # Assign nearest neighbor values to the invalid points
        confidence_array[nearby_invalid_coords[:, 0], nearby_invalid_coords[:, 1]] = valid_values[nearby_indices]
        """

    # Handle cells with no points
    confidence_array[np.isnan(confidence_array)] = min_confidence
    confidence_array[dem == dem_nodata] = np.nan

    return confidence_array
