""" Create a confidence map from a DEM and a point cloud."""
import numpy as np
import open3d as o3d
import rasterio
from scipy import ndimage
from tqdm import tqdm

def create_confidence_arr(dem: np.ndarray,
                          point_cloud,
                          transform,
                          interpolate: bool = False,
                          distance: int = 25,
                          no_data_val: int | float = -9999,
                          chunk_size:int = 5000,
                          buffer: int = 100,
                          min_confidence:int = 0
                          ):

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

    # Prepare output arrays
    confidence_array = np.zeros_like(dem, dtype=np.float32)
    count_array = np.zeros_like(dem, dtype=np.int16)

    # Extract x, y, and confidence from the point cloud
    x_coords = point_cloud[:, 0]
    y_coords = point_cloud[:, 1]
    confidence_values = point_cloud[:, 10]

    # release point cloud
    del point_cloud

    # Convert world coordinates to pixel indices
    inverse_transform = ~transform
    cols, rows = inverse_transform * (x_coords, y_coords)
    rows, cols = rows.astype(int), cols.astype(int)

    # release x and y coords
    del x_coords, y_coords

    # Create a mask to filter out points that fall outside the DEM bounds
    valid_mask = (rows >= 0) & (rows < dem.shape[0]) & (cols >= 0) & (cols < dem.shape[1])
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    confidence_values = confidence_values[valid_mask]

    # release valid mask
    del valid_mask

    # Use numpy indexing to accumulate confidence values and counts
    np.add.at(confidence_array, (rows, cols), confidence_values)
    np.add.at(count_array, (rows, cols), 1)

    # release rows and cols
    del rows, cols

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        confidence_array = np.divide(confidence_array, count_array, where=(count_array > 0))

    if interpolate:

        # get number of rows and columns
        rows_total, cols_total = confidence_array.shape
        result = confidence_array.copy()

        # Calculate total number of chunks for progress reporting.
        n_chunks_row = (rows_total + chunk_size - 1) // chunk_size
        n_chunks_col = (cols_total + chunk_size - 1) // chunk_size
        total_chunks = n_chunks_row * n_chunks_col

        # create progress bar
        pbar = tqdm(total=total_chunks, desc="Processing chunks")

        for row in range(0, rows_total, chunk_size):
            for col in range(0, cols_total, chunk_size):
                # Determine current chunk dimensions
                cur_rows = min(chunk_size, rows_total - row)
                cur_cols = min(chunk_size, cols_total - col)

                # Define window indices with buffer
                r_start = max(row - buffer, 0)
                c_start = max(col - buffer, 0)
                r_end = min(row + cur_rows + buffer, rows_total)
                c_end = min(col + cur_cols + buffer, cols_total)

                # Extract the chunk (with buffer)
                chunk_confidence = result[r_start:r_end, c_start:c_end]
                chunk_count = count_array[r_start:r_end, c_start:c_end]

                # Create a mask for missing points
                missing_mask = (chunk_count == 0)

                # Compute the distance transform for the chunk
                distances, indices = ndimage.distance_transform_edt(
                    missing_mask, return_distances=True, return_indices=True)
                nearest_row = indices[0]
                nearest_col = indices[1]

                # Determine where to fill: missing points within threshold
                threshold_mask = distances <= distance
                fill_mask = missing_mask & threshold_mask

                # Apply nearest neighbor interpolation for the fill region
                chunk_confidence[fill_mask] = chunk_confidence[nearest_row[fill_mask],
                nearest_col[fill_mask]]

                # Compute indices for the inner region (without buffer)
                inner_r_start = row - r_start
                inner_c_start = col - c_start
                inner_r_end = inner_r_start + cur_rows
                inner_c_end = inner_c_start + cur_cols

                # Write the processed inner region back to the result array
                result[row:row + cur_rows, col:col + cur_cols] = \
                    chunk_confidence[inner_r_start:inner_r_end, inner_c_start:inner_c_end]

                pbar.update(1)

    # Handle cells with no points
    confidence_array[np.isnan(confidence_array)] = min_confidence
    confidence_array[dem == no_data_val] = np.nan

    return confidence_array
