import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as Element_tree

from scipy.spatial.distance import cdist

import src.load.load_image as li
import src.export.export_ply as ep
import src.sfm_agi.snippets.adapt_frame as af
import src.sfm_agi.snippets.zip_folder as zp

debug_delete_files = False

def create_matching_structure(tp_dict, conf_dict,
                              project_files_folder, img_folder,
                              input_mode="agi",
                              min_tps=10, tolerance=1,
                              s_min=1, s_max=50,
                              camera_data=None):

    # mode can be 'agi' or 'bundler'
    if input_mode not in ['agi', 'bundler']:
        raise ValueError("Mode should be either 'agi' or 'bundler'")

    # create the path where the ply files are saved
    if input_mode == "agi":
        orig_pc_fld = os.path.join(project_files_folder, "0", "0", "point_cloud")
        path_pc_fld = os.path.join(orig_pc_fld, "point_cloud")
        if not os.path.exists(path_pc_fld):
            print("Create folder at ", path_pc_fld)
            os.makedirs(path_pc_fld)

    # init variables
    arr = None  # array to keep data of all matches
    image_dict = {}  # dict to store the loaded images
    image_dims = {}  # dict to store the image dimensions
    track_counter = 0  # counter for the tracks

    # sort dict by key
    tp_dict = dict(sorted(tp_dict.items()))
    conf_dict = dict(sorted(conf_dict.items()))

    # convert the image names to a sorted list and map them to an index
    image_names = sorted(set([key[0] for key in tp_dict.keys()] + [key[1] for key in tp_dict.keys()]))
    image_map = {name: idx for idx, name in enumerate(image_names)}

    # create one dataframe from different tp_dicts
    for key in tp_dict.keys():

        # get the tps and conf for a specific image pair
        tps = tp_dict[key]
        conf = conf_dict[key]

        # skip if there are too few tps
        if tps.shape[0] < min_tps:
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
            image_dims[img1_idx] = img1.shape[:2]
        else:
            img1 = image_dict[img1_id]
        if img2_id not in image_dict.keys():
            img2 = li.load_image(img2_id, img_folder)
            image_dict[img2_id] = img2
            image_dims[img2_idx] = img2.shape[:2]
        else:
            img2 = image_dict[img2_id]

        # get the points
        points1 = tps[:, :2]
        points2 = tps[:, 2:]

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

        # add the points to the array
        if arr is None:
            arr = np.vstack((points1, points2))
        else:
            arr = np.vstack((arr, points1, points2))

    # check that each track_id has a count of 2
    track_counts = np.unique(arr[:, 4], return_counts=True)
    for track_id, count in zip(track_counts[0], track_counts[1]):
        if count != 2:
            raise ValueError(f"Track {track_id} has {count} points")

    # convert to dataframe
    df = pd.DataFrame(arr)
    df.columns = ['x', 'y', 'image_idx', 'confidence', 'track_idx', 'color']

    # Get the x, y, and image_idx columns
    coords = df[['x', 'y']].values
    image_idx = df['image_idx'].values

    # Compute the pairwise Euclidean distances
    distances = cdist(coords, coords)

    # Find pairs of points with distances less than the tolerance
    close_pairs = np.where((distances < tolerance) & (distances > 0))  # Exclude distance of 0 (same point)

    # only keep pairs if the image_idx is identical for these pairs
    pairs = [(i, j) for i, j in zip(*close_pairs) if image_idx[i] == image_idx[j]]

    # Create a set to track processed indices
    processed = set()

    # Loop through matches and merge corresponding rows together
    for elem in pairs:

        # assure that matches only has two elements
        if len(elem) != 2:
            raise ValueError("Matches should only have two elements")

        # get the indices of the points
        i, j = elem

        # only merge if the points are not already processed
        if i not in processed and j not in processed:

            # get the track_idx for both points
            track_idx_i = df.iloc[i]['track_idx']
            track_idx_j = df.iloc[j]['track_idx']

            # Find the minimum track_idx between the two points
            min_track_idx = min(track_idx_i, track_idx_j)

            # get the average x, y
            avg_x = (df.iloc[i]['x'] + df.iloc[j]['x']) / 2
            avg_y = (df.iloc[i]['y'] + df.iloc[j]['y']) / 2

            # get the average color
            avg_color = int((df.iloc[i]['color'] + df.iloc[j]['color']) / 2)

            # update the coordinates at i and j
            df.at[i, 'x'] = avg_x
            df.at[j, 'x'] = avg_x
            df.at[i, 'y'] = avg_y
            df.at[j, 'y'] = avg_y

            # Update all entries with the same track_idx
            df.loc[df['track_idx'] == track_idx_i, 'color'] = avg_color
            df.loc[df['track_idx'] == track_idx_j, 'color'] = avg_color
            df.loc[df['track_idx'] == track_idx_i, 'track_idx'] = min_track_idx
            df.loc[df['track_idx'] == track_idx_j, 'track_idx'] = min_track_idx

            # Mark both points as processed to avoid updating them again
            processed.add(i)
            processed.add(j)

    # remove duplicates with the same value in image_idx and track_idx
    df = df.drop_duplicates(subset=['image_idx', 'track_idx'])

    if input_mode == "agi":

        # list to store the number of points per image
        points_per_image = []

        # create the p0 to px files
        for idx in image_map.values():

            # get the data for this particular image
            df_img = df[df['image_idx'] == idx]

            # only select certain columns and rename them
            df_img = df_img[['x', 'y', 'confidence', 'track_idx']]
            df_img.columns = ['x', 'y', 'size', 'id']

            # save the number of points per image
            points_per_image.append(df_img.shape[0])

            # create the path
            px_name = f"p{idx}.ply"
            px_path = os.path.join(path_pc_fld, px_name)

            # export the ply file
            ep.export_ply(df_img, px_path)

        # sort the dataframe by track_idx
        df = df.sort_values(by='track_idx')

        # get the unique track_idx and color
        df_tracks = df[['track_idx', 'color']].drop_duplicates()

        # get average color for each track
        df_tracks = df_tracks.groupby('track_idx').mean().reset_index()
        df_tracks = df_tracks[['color']].astype(int)

        # get number of all matches
        num_matches = df_tracks.shape[0]

        # create the tracks file
        tracks_path = os.path.join(path_pc_fld, "tracks.ply")
        ep.export_ply(df_tracks, tracks_path)

        # create the doc.xml file
        xml_path = os.path.join(path_pc_fld, "doc.xml")
        _create_doc_xml(xml_path, points_per_image, num_matches)

        # zip the folder
        output_zip_path = os.path.join(orig_pc_fld, 'point_cloud.zip')
        zp.zip_folder(path_pc_fld, output_zip_path, delete_files=debug_delete_files)

        # adapt the frame
        af.adapt_frame(project_files_folder, "point_cloud", "path", "point_cloud/point_cloud.zip")

    elif input_mode == "bundler":

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
        bundler_path = os.path.join(project_path_bundler_fld, "bundler.out")
        print(bundler_path)
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

            visited_tracks = set()

            # Step 3: Write point matches
            # iterate over the dataframe
            for i, row in df.iterrows():

                # get the track_idx of the row
                track_idx = row['track_idx']

                # skip if the track_idx is already visited
                if track_idx in visited_tracks:
                    continue
                else:
                    # add the track_idx to the visited tracks
                    visited_tracks.add(track_idx)

                # get all rows with the same track_idx
                match_array = df[df['track_idx'] == track_idx]

                # Dummy 3D point (since we only have 2D matches)
                f.write('0.0 0.0 0.0\n')  # XYZ coordinates of the point
                f.write(f'{int(row['color'])} {int(row['color'])} {int(row['color'])}\n')  # RGB color of the point (dummy value)

                fstr = f"{match_array.shape[0]} "
                for j, _row in match_array.iterrows():

                    # get image dimension
                    h, w = image_dims[int(_row['image_idx'])]
                    x_bundler = _row['x'] - w / 2
                    y_bundler = h / 2 - _row['y']

                    fstr += f"{int(_row['image_idx'])} "  # image index
                    fstr += f"{int(_row['track_idx'])} "  # track index
                    fstr += f"{x_bundler} {y_bundler} "  # x, y coordinates

                # remove last space and add newline
                fstr = fstr[:-1]
                f.write(fstr + "\n")

    else:
        raise ValueError("Mode should be either 'agi' or 'bundler'")


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
