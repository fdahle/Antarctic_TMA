import numpy as np
import open3d as o3d
import rasterio
from scipy import ndimage
from tqdm import tqdm

#from scipy.interpolate import griddata
#from scipy.spatial import cKDTree

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

    # delete pointcloud from memory
    del point_cloud

    # Create arrays to accumulate confidence values and counts
    count_array = np.zeros_like(dem, dtype=np.int16)

    # Invert the transform to map world coordinates to pixel coordinates
    inverse_transform = ~transform

    # Convert world coordinates to pixel indices (row, col)
    cols, rows = inverse_transform * (x_coords, y_coords)
    cols = cols.astype(int)
    rows = rows.astype(int)

    # delete x_coords and y_coords from memory
    del x_coords, y_coords

    # Create a mask to filter out points that fall outside the DEM bounds
    valid_mask = (rows >= 0) & (rows < dem.shape[0]) & (cols >= 0) & (cols < dem.shape[1])
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    confidence_values = confidence_values[valid_mask]

    # delete valid_mask from memory
    del valid_mask

    # Use numpy indexing to accumulate confidence values and counts
    np.add.at(confidence_array, (rows, cols), confidence_values)
    np.add.at(count_array, (rows, cols), 1)

    # delete rows and cols from memory
    del rows, cols

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        confidence_array = np.divide(confidence_array, count_array, where=(count_array > 0))

    if interpolate:
        print("Interpolating confidence values...")

        chunk_size = 5000
        buffer = 100

        rows_total, cols_total = confidence_array.shape
        result = confidence_array.copy()

        # Calculate total number of chunks for progress reporting.
        n_chunks_row = (rows_total + chunk_size - 1) // chunk_size
        n_chunks_col = (cols_total + chunk_size - 1) // chunk_size
        total_chunks = n_chunks_row * n_chunks_col

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
                # Here, inner region starts at:
                inner_r_start = row - r_start
                inner_c_start = col - c_start
                inner_r_end = inner_r_start + cur_rows
                inner_c_end = inner_c_start + cur_cols

                # Write the processed inner region back to the result array
                result[row:row + cur_rows, col:col + cur_cols] = \
                    chunk_confidence[inner_r_start:inner_r_end, inner_c_start:inner_c_end]

                pbar.update(1)

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
