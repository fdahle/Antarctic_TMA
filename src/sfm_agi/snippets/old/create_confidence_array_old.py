import numpy as np
import open3d as o3d
import rasterio

from collections import defaultdict
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


def create_confidence_arr_old(dem: np.ndarray,
                          point_cloud: np.ndarray,
                          transform,
                          interpolate: bool = False, distance: int = 10,
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

    dem[dem == dem_nodata] = np.nan

    # Initialize an array for confidence values, same shape as DEM
    confidence_array = np.zeros_like(dem, dtype=np.float32)
    count_array = np.zeros_like(dem, dtype=np.int32)

    # Extract X, Y, and confidence values from the point cloud
    x_coords = point_cloud[:, 0]
    y_coords = point_cloud[:, 1]
    confidence_values = point_cloud[:, 10]

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

    # Dictionary to accumulate confidence values for each (row, col)
    confidence_accumulator = defaultdict(list)

    # close possible open tqdm instances
    tqdm._instances.clear()  # noqa

    # Accumulate confidence values for each valid (row, col)
    for idx, (row, col, confidence) in (pbar := tqdm(enumerate(zip(valid_rows, valid_cols, valid_confidences)), total=len(valid_rows))):

        # Update progress bar
        pbar.set_description("Accumulate confidence values")
        pbar.set_postfix_str(f"row {row}, col {col} with confidence {confidence}")

        # Append confidence value to the list
        confidence_accumulator[(row, col)].append(confidence)

        # Update progress bar for the last iteration
        if idx == len(valid_rows) - 1:
            pbar.set_postfix_str("Finished!")

    # close possible open tqdm instances
    tqdm._instances.clear()  # noqa

    # Now compute the average confidence for each cell
    for idx, (key, confidences) in (pbar := tqdm(enumerate(confidence_accumulator.items()),
                                                 total=len(confidence_accumulator))):
        row, col = key

        # Update progress bar
        pbar.set_description("Compute average confidence")
        pbar.set_postfix_str(f"row {row}, col {col}")

        # Compute the average confidence value
        confidence_array[row, col] = np.mean(confidences)
        count_array[row, col] = len(confidences)

        # Update progress bar for the last iteration
        if idx == len(confidence_accumulator) - 1:
            pbar.set_postfix_str("Finished!")

    if interpolate:

        # Create mask for valid values
        valid_mask = count_array > 0
        invalid_mask = ~valid_mask

        # Get coordinates of valid and invalid points
        valid_coords = np.argwhere(valid_mask)
        invalid_coords = np.argwhere(invalid_mask)

        # Values at valid coordinates
        valid_values = confidence_array[valid_mask]

        # Create a KDTree for efficient distance calculation
        tree = cKDTree(valid_coords)

        # Find the distance to the nearest valid point for each invalid point
        distances, _ = tree.query(invalid_coords)

        # Mask for points within the distance threshold
        nearby_mask = distances <= distance

        # Interpolate using valid points only for the nearby invalid points
        nearby_invalid_coords = invalid_coords[nearby_mask]  # Only interpolate close invalid points
        confidence_filled = griddata(
            valid_coords,
            valid_values,
            nearby_invalid_coords,
            method='linear',
            fill_value=np.nan  # Optional: fill distant points with NaN
        )

        # Now assign the interpolated values back to the original confidence array
        confidence_array[nearby_invalid_coords[:, 0], nearby_invalid_coords[:, 1]] = confidence_filled

    # Handle cells with no points
    confidence_array[np.isnan(confidence_array)] = min_confidence
    confidence_array[np.isnan(dem)] = np.nan

    return confidence_array
