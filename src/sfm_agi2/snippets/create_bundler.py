"""Create a Bundler-compatible output file"""

# Library imports
import numpy as np
import os
import pandas as pd
from io import StringIO
from numba import njit
from sklearn.neighbors import KDTree
from tqdm import tqdm

# Local imports
import src.load.load_image as li

# TODO FIX CAMERA DATA

def create_bundler(img_folder: str,
                   bundler_folder: str,
                   tp_dict: dict[tuple[str, str], np.ndarray],
                   conf_dict: dict[tuple[str, str], np.ndarray],
                   camera_data: dict[str, ] | None = None,
                   min_tps: int=0,
                   s_min:int = 1, s_max: int = 50,
                   tolerance: int = 1):
    """
    Create a Bundler-compatible output file from tie-points and confidence values.

    This function merges matched points between images, assigns consistent
    track IDs, and saves the result in the Bundler format. The marker sizes
    are equivalent to the confidence values.

    Args:
        img_folder (str): Folder containing source images (.tif).
        tp_dict (Dict[Tuple[str, str], np.ndarray]): Mapping of image pairs
            to Nx4 arrays of matched points. Each array must be [x1, y1, x2, y2].
        conf_dict (Dict[Tuple[str, str], np.ndarray]): Mapping of image pairs
            to Nx1 confidence arrays.
        min_tps (int, optional): Minimum number of tie-points required per image pair. Defaults to 0.
        s_min (int, optional): Minimum size of a marker in the bundler.
            Defaults to 1.
        s_max (int, optional): Maximum size of a marker in the bundler.
            Defaults to 50.
        tolerance (float, optional): Pixel distance for merging close-by tie-points.
            Defaults to 1.
        camera_data (Optional[list], optional): List of camera calibration dictionaries. Dummy values are used if None.

    Returns:
        str: Path to the saved `bundler.out` file.
    """

    track_counter = 0  # global counter for track indices
    image_dict = {}    # cache for loaded images
    image_dims = {}    # dictionary to store image dimensions

    # Ensure dictionaries are sorted by key for consistency
    tp_dict = dict(sorted(tp_dict.items()))
    conf_dict = dict(sorted(conf_dict.items()))

    # Create a sorted list of image names (without extension) and map them to indices
    image_names = sorted([os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.endswith(".tif")])
    image_map = {name: idx for idx, name in enumerate(image_names)}

    # Helper function for loading images with caching
    def load_image_cached(image_id: str):
        """load image from disk and cache it"""
        if image_id not in image_dict:
            img = li.load_image(image_id, img_folder)
            image_dict[image_id] = img
            image_dims[image_map[image_id]] = (img.shape[1], img.shape[0])
        return image_dict[image_id]

    # List to collect rows before converting to DataFrame
    all_rows = []

    # create progress bar
    pbar = tqdm(total=len(tp_dict.keys()), desc="Collect tie-points for bundler",
                position=0, leave=True)

    # Process each image pair from tp_dict
    for key in tp_dict.keys():

        pbar.update(1)

        # Unpack image IDs and retrieve corresponding data
        img1_id, img2_id = key
        tps = tp_dict[key]
        conf = conf_dict[key]

        # Skip if there are too few tie-points
        if tps.shape[0] < min_tps:
            continue

        # get image ids and their indices
        img1 = load_image_cached(img1_id)
        img2 = load_image_cached(img2_id)
        img1_idx = image_map[img1_id]
        img2_idx = image_map[img2_id]

        # get the points
        points1 = tps[:, :2]
        points2 = tps[:, 2:]

        if np.amax(points1[:, 0]) > img1.shape[1] or np.amax(points1[:, 1]) > img1.shape[0]:
            print(f"{img1_id}, {img2_id} - Image 1 shape: {img1.shape}, "
                  f"max (x,y): ({np.amax(points1[:, 0])}, {np.amax(points1[:, 1])})")
            raise ValueError("Points outside the image for image 1.")
        if np.amax(points2[:, 0]) > img2.shape[1] or np.amax(points2[:, 1]) > img2.shape[0]:
            print(f"{img2_id}, {img2_id} - Image 2 shape: {img2.shape}, "
                  f"max (x,y): ({np.amax(points2[:, 0])}, {np.amax(points2[:, 1])})")
            raise ValueError("Points outside the image for image 2.")

        # Ensure confidence is a column vector and convert it to point marker sizes
        conf = conf.reshape(-1, 1)
        conf = np.clip(conf, 0, 1)  # Confidence values between 0 and 1
        conf_size = s_min + (s_max - s_min) * (1 - conf)

        # Create unique track indices for these points
        tracks = np.arange(points1.shape[0]).reshape(-1, 1) + track_counter
        track_counter += points1.shape[0]

        # Convert points to integer indices for pixel value extraction
        points1_indices = points1.astype(int)
        points2_indices = points2.astype(int)
        pixel_values1 = img1[points1_indices[:, 1], points1_indices[:, 0]]
        pixel_values2 = img2[points2_indices[:, 1], points2_indices[:, 0]]

        # Append additional information: image index, marker size, track index, and pixel value
        points1 = np.hstack((points1, np.ones((points1.shape[0], 1)) * img1_idx,
                             conf_size, tracks, pixel_values1.reshape(-1, 1)))
        points2 = np.hstack((points2, np.ones((points2.shape[0], 1)) * img2_idx,
                             conf_size, tracks, pixel_values2.reshape(-1, 1)))

        # extend the all_rows list
        all_rows.extend(points1.tolist())
        all_rows.extend(points2.tolist())

    # close the progress bar
    pbar.close()

    # Create DataFrame with clear column names
    df = pd.DataFrame(all_rows, columns=['x', 'y', 'image_idx',
                                         'confidence', 'track_idx', 'color'])

    # Verify that each original track appears exactly twice
    track_id_counts = df.groupby('track_idx').size()
    if (track_id_counts != 2).any():
        raise ValueError("Some track_ids do not appear exactly 2 times.")

    # create progress bar
    pbar = tqdm(total=len(image_map.keys()), desc="Merge tracks for bundler",
                position=0, leave=True)

    # Merge tracks across images using KDTree for each image
    processed = set()
    for image_key, image_idx in image_map.items():

        pbar.update(1)

        df_sub = df[df['image_idx'] == image_idx].copy()
        coords = df_sub[['x', 'y']].values
        if coords.shape[0] == 0:
            continue

        # For each point, find neighbors within tolerance
        tree = KDTree(coords)
        indices = tree.query_radius(coords, r=tolerance)
        pairs = [(i, j) for i, neighbors in enumerate(indices) for j in neighbors if i != j]
        pairs = [(i, j) for i, j in pairs if i not in processed and j not in processed]

        if len(pairs) == 0:
            continue

        # Pre-extract columns for speed and merge tracks using the helper function
        track_idx_arr = df_sub['track_idx'].values
        color_arr = df_sub['color'].values
        x_arr = df_sub['x'].values
        y_arr = df_sub['y'].values

        # _merge_points is assumed to be a pre-defined, optimized function (e.g., using Numba)
        x_arr, y_arr, color_arr, track_idx_arr, track_idx_mapping = _merge_points(pairs, x_arr, y_arr,
                                                                                  color_arr, track_idx_arr)
        # Update the dataframe subset
        df_sub['x'] = x_arr
        df_sub['y'] = y_arr
        df_sub['color'] = color_arr
        df_sub['track_idx'] = track_idx_arr
        df.loc[df['image_idx'] == image_idx, ['x', 'y', 'color', 'track_idx']] = df_sub[
            ['x', 'y', 'color', 'track_idx']]

        # Apply track index mapping across the whole dataframe
        for max_track_idx, min_track_idx in track_idx_mapping.items():
            df.loc[df['track_idx'] == max_track_idx, 'track_idx'] = min_track_idx
            processed.add(min_track_idx)

    pbar.close()

    # Remove duplicate track entries per image and ensure integer type for indices
    df = df.drop_duplicates(subset=['image_idx', 'track_idx'])
    df['image_idx'] = df['image_idx'].astype(int)
    df['track_idx'] = df['track_idx'].astype(int)

    # Final verification: Each track must appear in at least 2 images
    track_id_counts = df.groupby('track_idx').size()
    if (track_id_counts < 2).any():
        raise ValueError("Error: Some track_ids do not appear in at least 2 images.")

    # Create dummy camera data if not provided
    if camera_data is None:
        camera_data = [{
            "focal_length": 1.0,
            "k1": 0.0,
            "k2": 0.0,
            "rotation_matrix": [1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0],
            "translation_vector": [0.0, 0.0, 0.0]
        } for _ in image_map]

    # Calculate total number of input tie-points (before merging)
    # num_points = sum([matches.shape[0] for matches in tp_dict.values()])
    num_points = df['track_idx'].nunique()

    # Ensure bundler folder exists
    bundler_path = os.path.join(bundler_folder, "bundler.out")

    # Add image dimensions and calculate bundler coordinates
    df['image_dims_x'] = df['image_idx'].apply(lambda x: image_dims[x][0])
    df['image_dims_y'] = df['image_idx'].apply(lambda x: image_dims[x][1])

    df['bundler_x'] = df['x'] - df['image_dims_x'] / 2
    df['bundler_y'] = df['image_dims_y'] / 2 - df['y']

    # Remove gaps in track indices by mapping them sequentially
    unique_track_ids = sorted(df['track_idx'].unique())
    sequential_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_track_ids)}
    df['track_idx'] = df['track_idx'].map(sequential_mapping)

    # Group by track for bundler file writing
    df_grouped = df.groupby('track_idx')

    # Write bundler file using a buffer for performance
    with open(bundler_path, 'w') as f:
        # Write header
        f.write('# Bundle file v0.3\n')
        f.write(f'{len(camera_data)} {num_points}\n')

        # Write camera data
        for cam in camera_data:
            f.write(f'{cam["focal_length"]} {cam["k1"]} {cam["k2"]}\n')
            f.write(f'{cam["rotation_matrix"][0]} {cam["rotation_matrix"][1]} {cam["rotation_matrix"][2]}\n')
            f.write(f'{cam["rotation_matrix"][3]} {cam["rotation_matrix"][4]} {cam["rotation_matrix"][5]}\n')
            f.write(f'{cam["rotation_matrix"][6]} {cam["rotation_matrix"][7]} {cam["rotation_matrix"][8]}\n')
            f.write(' '.join(str(val) for val in cam["translation_vector"]) + "\n")

        # creat the buffer for the bundler file
        buffer = StringIO()

        # create progress bar
        pbar = tqdm(total=len(df_grouped), desc="Write bundler file",
                    position=0, leave=True)

        for track_idx, group in df_grouped:

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix_str("Track: {}".format(track_idx))

            # Average color for the track (dummy value)
            avg_color = int(round(group['color'].mean()))

            # Dummy 3D point (since only 2D data is available)
            buffer.write('0.0 0.0 0.0\n')
            buffer.write(f'{avg_color} {avg_color} {avg_color}\n')

            # Build the string for the number of images and each observation
            obs_parts = []
            for _, row in group.iterrows():
                obs_parts.extend([str(int(row['image_idx'])),
                                  str(int(row['track_idx'])),
                                  f"{row['bundler_x']}",
                                  f"{row['bundler_y']}"])
            obs_line = f"{group.shape[0]} " + " ".join(obs_parts)
            buffer.write(obs_line + "\n")

        # close the progress bar
        pbar.set_postfix_str("- Finished -")
        pbar.close()

        # Write the buffered content to file
        f.write(buffer.getvalue())

    return bundler_path

@njit
def _merge_points(pairs, x_arr, y_arr, color_arr, track_idx_arr):
    """ Merges points in pairs and updates their x, y, color, and track_idx """
    processed = set()
    track_idx_mapping = {}

    for i, j in pairs:

        # Get track_idx and color for both points
        track_idx_i = track_idx_arr[i]
        track_idx_j = track_idx_arr[j]

        if track_idx_i in processed or track_idx_j in processed:
            continue

        # Find the minimum and maximum track_idx between the two points
        min_track_idx = min(track_idx_i, track_idx_j)
        max_track_idx = max(track_idx_i, track_idx_j)

        # Compute the average x, y, and color values
        avg_x = (x_arr[i] + x_arr[j]) / 2
        avg_y = (y_arr[i] + y_arr[j]) / 2
        avg_color = (color_arr[i] + color_arr[j]) // 2  # integer division

        # Update coordinates and color for both points
        x_arr[i] = x_arr[j] = avg_x
        y_arr[i] = y_arr[j] = avg_y
        color_arr[i] = color_arr[j] = avg_color

        # Update track_idx for both points
        track_idx_arr[i] = track_idx_arr[j] = min_track_idx

        # Record the mapping from max_track_idx to min_track_idx
        track_idx_mapping[max_track_idx] = min_track_idx

        # Mark points as processed
        processed.add(track_idx_i)
        processed.add(track_idx_j)

    return x_arr, y_arr, color_arr, track_idx_arr, track_idx_mapping
