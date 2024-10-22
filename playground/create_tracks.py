import numpy as np


def create_tracks(tp_dict, conf_dict):
    """
    Create tracks from tie-points and confidence values.

    Parameters:
    - tp_dict: A dictionary where keys are 'img1id_img2id' strings, and values are numpy arrays of shape (N, 4),
               containing x1, y1, x2, y2 coordinates.
    - conf_dict: A dictionary with the same keys as tp_dict, where values are numpy arrays of confidence values.

    Returns:
    - track_list: A list of tracks. Each track is a dictionary mapping from image IDs to (x, y) coordinates.
    """
    # Initialize data structures
    image_keypoints = {}  # {img_id: {quantized (x, y): kp_id}}
    kp_counter = {}  # {img_id: current kp_id}
    parent = {}  # Union-Find parent
    rank = {}  # Union-Find rank

    # Union-Find functions
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        else:
            parent[v_root] = u_root
            if rank[u_root] == rank[v_root]:
                rank[u_root] += 1

    # Function to quantize coordinates to handle floating-point precision
    def quantize_coord(x, y, precision=1e-3):
        return (int(round(x / precision)), int(round(y / precision)))

    # Process each pair of images
    for key in tp_dict:
        # Parse image IDs from the key
        img1id, img2id = key.split('_')

        # Extract tie-points
        x1y1x2y2 = tp_dict[key]  # Shape: (N, 4)
        x1 = x1y1x2y2[:, 0]
        y1 = x1y1x2y2[:, 1]
        x2 = x1y1x2y2[:, 2]
        y2 = x1y1x2y2[:, 3]
        N = x1.shape[0]

        # Process each matched pair of key-points
        for k in range(N):
            x1k, y1k = x1[k], y1[k]
            x2k, y2k = x2[k], y2[k]

            # Quantize coordinates
            qx1k, qy1k = quantize_coord(x1k, y1k)
            qx2k, qy2k = quantize_coord(x2k, y2k)

            # Handle image 1 key-points
            if img1id not in image_keypoints:
                image_keypoints[img1id] = {}
                kp_counter[img1id] = 0
            kp_dict1 = image_keypoints[img1id]
            if (qx1k, qy1k) not in kp_dict1:
                kp_id1 = kp_counter[img1id]
                kp_dict1[(qx1k, qy1k)] = kp_id1
                kp_counter[img1id] += 1
            else:
                kp_id1 = kp_dict1[(qx1k, qy1k)]

            # Handle image 2 key-points
            if img2id not in image_keypoints:
                image_keypoints[img2id] = {}
                kp_counter[img2id] = 0
            kp_dict2 = image_keypoints[img2id]
            if (qx2k, qy2k) not in kp_dict2:
                kp_id2 = kp_counter[img2id]
                kp_dict2[(qx2k, qy2k)] = kp_id2
                kp_counter[img2id] += 1
            else:
                kp_id2 = kp_dict2[(qx2k, qy2k)]

            # Create unique identifiers for key-points
            kp1 = (img1id, kp_id1)
            kp2 = (img2id, kp_id2)

            # Initialize Union-Find structures
            for kp in [kp1, kp2]:
                if kp not in parent:
                    parent[kp] = kp
                    rank[kp] = 0
            # Union the key-points
            union(kp1, kp2)

    # Collect tracks from Union-Find structures
    tracks = {}  # root -> list of key-points
    for kp in parent:
        root = find(kp)
        if root not in tracks:
            tracks[root] = []
        tracks[root].append(kp)

    # Convert tracks to the desired format
    track_list = []
    for track in tracks.values():
        track_dict = {}
        for img_id, kp_id in track:
            # Retrieve the quantized coordinates
            kp_dict = image_keypoints[img_id]
            for (qxy), id in kp_dict.items():
                if id == kp_id:
                    # Convert quantized coordinates back to floating-point values
                    x = qxy[0] * 1e-3
                    y = qxy[1] * 1e-3
                    track_dict[img_id] = (x, y)
                    break
        track_list.append(track_dict)

    return track_list


if __name__ == "__main__":

    import os
    import src.sfm_agi.snippets.find_tie_points_for_sfm as ftpfs

    project_fld = "/data/ATM/data_1/sfm/agi_projects/"
    project_name = "another_matching_try_agi"

    # create the path to the project
    project_path = os.path.join(project_fld, project_name, project_name + ".psx")

    img_folder = os.path.join(project_fld, project_name, 'data', 'images')
    mask_folder = os.path.join(project_fld, project_name, 'data', 'masks_adapted')

    tp_dict, conf_dict = ftpfs.find_tie_points_for_sfm(img_folder, mask_folder, "sequential")

    tracks = create_tracks(tp_dict, conf_dict)
    print(tracks)