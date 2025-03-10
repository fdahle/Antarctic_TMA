# Python imports
import os
import xml.etree.ElementTree as Element_tree
from io import StringIO

# Library imports
import numpy as np
import pandas as pd
from numba import njit
from sklearn.neighbors import KDTree
from tqdm import tqdm

# Local imports
import src.load.load_image as li

debug_delete_files = False

def create_matching_structure(tp_dict, conf_dict,
                              project_files_folder, img_folder,
                              min_tps=10, tolerance=1,
                              s_min=1, s_max=50,
                              camera_data=None):
    """

    Args:
        tp_dict:
        conf_dict:
        project_files_folder:
        img_folder:
        min_tps:
        tolerance: tolerance for the distance between points to be considered as a match (in pixels)
        s_min:
        s_max:
        camera_data:

    Returns:

    """

    # init variables
    track_counter = 0  # counter for the tracks
    image_dict = {}  # dict to store the loaded images
    image_dims = {}  # dict to store the image dimensions

    # sort dict by key
    tp_dict = dict(sorted(tp_dict.items()))
    conf_dict = dict(sorted(conf_dict.items()))

    # convert the image names to a sorted list and map them to an index
    image_names = [f.split(".")[0] for f in os.listdir(img_folder) if f.endswith(".tif")]
    image_names = sorted(image_names)
    image_map = {name: idx for idx, name in enumerate(image_names)}

    # List to collect rows before converting them to a DataFrame
    all_rows = []

    # add the points to the dataframe
    for idx, key in (pbar := tqdm(enumerate(tp_dict.keys()), total=len(tp_dict))):

        # set description and postfix
        pbar.set_description("Add tie_points to dataframe")
        pbar.set_postfix_str(f"{key[0]} and {key[1]}")

        # get the tps and conf for this specific image pair
        tps = tp_dict[key]
        conf = conf_dict[key]

        # skip if there are too few tps
        if tps.shape[0] < min_tps:
            if idx == len(tp_dict) - 1:
                pbar.set_postfix_str("Finished!")
            continue

        # get the two image ids
        img1_id = key[0]
        img2_id = key[1]

        # get the image index
        img1_idx = image_map[img1_id]
        img2_idx = image_map[img2_id]

        # check if the image is already loaded
        if img1_id not in image_dict.keys():
            img1 = li.load_image(img1_id, img_folder)
            image_dict[img1_id] = img1
            image_dims[img1_idx] = (img1.shape[1], img1.shape[0])
        else:
            img1 = image_dict[img1_id]
        if img2_id not in image_dict.keys():
            img2 = li.load_image(img2_id, img_folder)
            image_dict[img2_id] = img2
            image_dims[img2_idx] = (img2.shape[1], img2.shape[0])
        else:
            img2 = image_dict[img2_id]

        # get the points
        points1 = tps[:, :2]
        points2 = tps[:, 2:]

        if np.amax(points1[:, 0]) > img1.shape[1] or np.amax(points1[:, 1]) > img1.shape[0]:
            print(img1_id, img2_id)
            print(f"Image 1 shape: {img1.shape}")
            print(f"Max y: {np.amax(points1[:, 1])}, Max x: {np.amax(points1[:, 0])}")
            raise ValueError("Points outside the image.")
        if np.amax(points2[:, 0]) > img2.shape[1] or np.amax(points2[:, 1]) > img2.shape[0]:
            print(img2_id, img2_id)
            print(f"Image 2 shape: {img2.shape}")
            print(f"Max y: {np.amax(points2[:, 1])}, Max x: {np.amax(points2[:, 0])}")
            raise ValueError("Points outside the image.")

        # ensure conf is a column vector for stacking
        conf = conf.reshape(-1, 1)

        # convert conf to size
        conf = np.clip(conf, 0, 1)  # Ensure confidence values are clipped between 0 and 1
        conf_size = s_min + (s_max - s_min) * (1 - conf)

        # create tracks for the points
        tracks = np.arange(points1.shape[0]).reshape(-1, 1)
        tracks = tracks + track_counter
        track_counter = track_counter + points1.shape[0]

        # Convert x, y to integer indices for image access
        points1_indices = points1.astype(int)
        points2_indices = points2.astype(int)

        # Get the image pixel values
        pixel_values1 = img1[points1_indices[:, 1], points1_indices[:, 0]]
        pixel_values2 = img2[points2_indices[:, 1], points2_indices[:, 0]]

        # add image_idx to the points
        points1 = np.hstack((points1, np.ones((points1.shape[0], 1)) * img1_idx))
        points2 = np.hstack((points2, np.ones((points2.shape[0], 1)) * img2_idx))

        # add confidence to the points
        points1 = np.hstack((points1, conf_size))
        points2 = np.hstack((points2, conf_size))

        # add the tracks to the points
        points1 = np.hstack((points1, tracks))
        points2 = np.hstack((points2, tracks))

        # add the pixel values to the points
        points1 = np.hstack((points1, pixel_values1.reshape(-1, 1)))
        points2 = np.hstack((points2, pixel_values2.reshape(-1, 1)))

        # extend the all_rows list
        all_rows.extend(points1.tolist())
        all_rows.extend(points2.tolist())

        # update the progress bar
        if idx == len(tp_dict) - 1:
            pbar.set_postfix_str("Finished!")

    # Create DataFrame from the collected data
    df = pd.DataFrame(all_rows, columns=['x', 'y', 'image_idx', 'confidence', 'track_idx', 'color'])

    # check that each track_id has a count of 2
    track_id_counts = df.groupby('track_idx').size()
    if (track_id_counts != 2).any():
        print("Some track_ids do not appear exactly 2 times.")

    # Create a set to track processed indices
    processed = set()

    # iterate the image_idx to find tracks over multiple images
    for idx, image_key in (pbar := tqdm(enumerate(image_map.keys()), total=len(image_map))):

        pbar.set_description("Merge tracks")
        pbar.set_postfix_str(f"Image {image_key}")

        # get the idx
        image_idx = image_map[image_key]

        # get subset of df
        df_sub = df[df['image_idx'] == image_idx].copy()

        # Get coords columns
        coords = df_sub[['x', 'y']].values

        # assure that there are any points in the image
        if coords.shape[0] == 0:
            continue

        # Use KDTree for efficient neighbor search within the tolerance
        tree = KDTree(coords)

        # Find indices of points within the given tolerance
        indices = tree.query_radius(coords, r=tolerance)

        # Convert the result to a list of pairs (i, j)
        pairs = [(i, j) for i, neighbors in enumerate(indices) for j in neighbors if i != j]

        # remove pairs where one of the points is already processed
        pairs = [(i, j) for i, j in pairs if i not in processed and j not in processed]

        # Skip if there are no pairs
        if len(pairs) == 0:
            continue

        # Pre-extract necessary columns as numpy arrays for fast in-loop access
        track_idx_arr = df_sub['track_idx'].values
        color_arr = df_sub['color'].values
        x_arr = df_sub['x'].values
        y_arr = df_sub['y'].values

        # Use Numba to process the pairs and update arrays
        x_arr, y_arr, color_arr, track_idx_arr, track_idx_mapping = _merge_points(pairs, x_arr, y_arr,
                                                                                  color_arr, track_idx_arr)

        # Put the updated values back into df_sub
        df_sub['x'] = x_arr
        df_sub['y'] = y_arr
        df_sub['color'] = color_arr
        df_sub['track_idx'] = track_idx_arr

        # Put the updated df_sub back into the original df
        df.loc[df['image_idx'] == image_idx, ['x', 'y', 'color', 'track_idx']] = df_sub[
            ['x', 'y', 'color', 'track_idx']]

        # update the tracking indices for the other elements as well
        for max_track_idx, min_track_idx in track_idx_mapping.items():
            df.loc[df['track_idx'] == max_track_idx, 'track_idx'] = min_track_idx

            # add min_track_idx to processed
            processed.add(min_track_idx)

        if idx == len(image_map) - 1:
            pbar.set_postfix_str("Finished!")

    # remove duplicates with the same value in image_idx and track_idx
    df = df.drop_duplicates(subset=['image_idx', 'track_idx'])

    # set image_idx and track_idx to integer
    df['image_idx'] = df['image_idx'].astype(int)
    df['track_idx'] = df['track_idx'].astype(int)

    # check that all track_ids appear at least 2 times
    track_id_counts = df.groupby('track_idx').size()
    if (track_id_counts < 2).any():
        raise ValueError("Some track_ids do not appear at least 2 times.")

    # replace with dummy data if camera_data is None:
    if camera_data is None:

        camera_data = []
        # create dummy camera data
        for _ in image_map.keys():
            camera_data.append({
                "focal_length": 1.0,
                "k1": 0.0,
                "k2": 0.0,
                "rotation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "translation_vector": [0.0, 0.0, 0.0]
            })

    # get number of points
    num_points = sum([matches.shape[0] for matches in tp_dict.values()])

    # create fld for the bundler file
    project_path_data_fld = os.path.abspath(os.path.join(project_files_folder, '..'))
    project_path_bundler_fld = os.path.join(project_path_data_fld, "data", "bundler")
    if os.path.isdir(project_path_bundler_fld) is False:
        os.makedirs(project_path_bundler_fld)

    # define the path to the bundler file
    bundler_path = os.path.join(project_path_bundler_fld, "bundler.out")

    # add the image dimensions (split in x and y) to the df
    df['image_dims_x'] = df['image_idx'].apply(lambda x: image_dims[x][0])
    df['image_dims_y'] = df['image_idx'].apply(lambda x: image_dims[x][1])

    # calculate bundler_x and bundler_y
    df['bundler_x'] = df['x'] - df['image_dims_x'] / 2
    df['bundler_y'] = df['image_dims_y'] / 2 - df['y']

    # remove gaps in the track_idx
    unique_track_ids = sorted(df['track_idx'].unique())
    sequential_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_track_ids)}
    df['track_idx'] = df['track_idx'].map(sequential_mapping)

    # group the df by track_idx
    df_grouped = df.groupby('track_idx')

    # write the bundler file
    with open(bundler_path, 'w') as f:

        # Step 1: Write header
        f.write('# Bundle file v0.3\n')
        f.write(f'{len(camera_data)} {num_points}\n')  # Number of cameras and number of points (we'll update later)

        # Step 2: Write camera data
        for cam in camera_data:
            f.write(f'{cam["focal_length"]} {cam["k1"]} {cam["k2"]}\n')  # focal length, distortion coefficients
            f.write(f'{cam["rotation_matrix"][0]} {cam["rotation_matrix"][1]} {cam["rotation_matrix"][2]}\n')
            f.write(f'{cam["rotation_matrix"][3]} {cam["rotation_matrix"][4]} {cam["rotation_matrix"][5]}\n')
            f.write(f'{cam["rotation_matrix"][6]} {cam["rotation_matrix"][7]} {cam["rotation_matrix"][8]}\n')
            f.write(f'{cam["translation_vector"][0]} {cam["translation_vector"][1]} {cam["translation_vector"][2]}\n')

        # Buffer for bulk writing
        buffer = StringIO()

        # Step 3: Write point matches
        # iterate over the dataframe
        for track_idx, group in (pbar := tqdm(df_grouped, total=df_grouped.ngroups)):

            # set progress bar description and postfix
            pbar.set_description("Write tracks to bundler file")
            pbar.set_postfix_str(f"Track: {track_idx}")

            # calculate average color
            avg_color = int(group['color'].mean())

            # Dummy 3D point (since we only have 2D matches)
            buffer.write('0.0 0.0 0.0\n')  # XYZ coordinates of the point
            buffer.write(f'{avg_color} {avg_color} {avg_color}\n')  # RGB color of the point (dummy value)

            # Number of images in which the point appears
            fstr = f"{group.shape[0]} "

            # Iterate over the rows in the group
            for _, _row in group.iterrows():
                fstr += f"{int(_row['image_idx'])} "  # image index
                fstr += f"{int(_row['track_idx'])} "  # track index
                fstr += f"{_row['bundler_x']} {_row['bundler_y']} "  # x, y coordinates

            # remove last space and add newline
            fstr = fstr[:-1]
            buffer.write(fstr + "\n")

            # Update the progress bar when finished
            if track_idx == df_grouped.ngroups - 1:
                pbar.set_postfix_str("Finished!")

        # Write all the buffered content at once to the file
        f.write(buffer.getvalue())

def _create_doc_xml(xml_path, points_per_image, num_matches):
    # Create the root element
    root = Element_tree.Element("point_cloud", version="1.2.0")

    # Create the params element
    params = Element_tree.SubElement(root, "params")

    # Add dataType element
    Element_tree.SubElement(params, "dataType").text = "uint8"

    # Add bands element
    bands = Element_tree.SubElement(params, "bands")
    Element_tree.SubElement(bands, "band")

    # Add tracks element
    Element_tree.SubElement(root, "tracks", path="tracks.ply", count=str(num_matches))

    # counter for the camera ply files
    c_counter = 0

    # Add projections elements
    for elem in points_per_image:

        Element_tree.SubElement(root, "projections", camera_id=str(c_counter),
                                path=f"p{c_counter}.ply",
                                count=str(elem))

        c_counter += 1

    # TODO: FIX METADATA
    metadata = {
        "Info/LastSavedDateTime": "2024:08:19 16:30:19",
        "Info/LastSavedSoftwareVersion": "2.1.2.18358",
        "Info/OriginalDateTime": "2024:08:19 16:30:19",
        "Info/OriginalSoftwareVersion": "2.1.2.18358",
        "MatchPhotos/cameras": "",
        "MatchPhotos/descriptor_type": "binary",
        "MatchPhotos/descriptor_version": "1.1.0",
        "MatchPhotos/downscale": "1",
        "MatchPhotos/downscale_3d": "1",
        "MatchPhotos/duration": "2.523574",
        "MatchPhotos/filter_mask": "true",
        "MatchPhotos/filter_stationary_points": "true",
        "MatchPhotos/generic_preselection": "true",
        "MatchPhotos/guided_matching": "false",
        "MatchPhotos/keep_keypoints": "true",
        "MatchPhotos/keypoint_limit": "40000",
        "MatchPhotos/keypoint_limit_3d": "100000",
        "MatchPhotos/keypoint_limit_per_mpx": "1000",
        "MatchPhotos/laser_scans_vertical_axis": "0",
        "MatchPhotos/mask_tiepoints": "true",
        "MatchPhotos/match_laser_scans": "false",
        "MatchPhotos/max_workgroup_size": "100",
        "MatchPhotos/ram_used": "349401088",
        "MatchPhotos/reference_preselection": "true",
        "MatchPhotos/reference_preselection_mode": "0",
        "MatchPhotos/reset_matches": "true",
        "MatchPhotos/subdivide_task": "true",
        "MatchPhotos/tiepoint_limit": "4000",
        "MatchPhotos/workitem_size_cameras": "20",
        "MatchPhotos/workitem_size_pairs": "80",
    }

    # Add the metadata
    meta = Element_tree.SubElement(root, "meta")
    for key, value in metadata.items():
        Element_tree.SubElement(meta, "property", name=key, value=value)

    # Convert the XML structure to a string
    xml_str = Element_tree.tostring(root, encoding='unicode', method='xml')

    # Add XML declaration at the beginning
    xml_declaration = '<?xml version="1.0"?>\n'

    # Combine the XML declaration and the generated XML string
    full_xml = xml_declaration + xml_str

    with open(xml_path, "w") as f:
        f.write(full_xml)



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
