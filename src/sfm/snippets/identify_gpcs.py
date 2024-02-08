import copy
import numpy as np
import torch

from tqdm import tqdm

import src.base.find_tie_points as ftp

import src.display.display_images as di


from external.lightglue import SuperPoint

def identify_gcps(images, transforms):

    # init tie-point matcher
    tp_finder = ftp.TiePointDetector(matching_method="lightglue")

    dict_gcps = {}
    dict_conf = {}

    batch_size = 100

    # iterate all images
    for i, image in enumerate(images):

        print("transform image", i)

        transform = transforms[i]

        # find interesting point in the images
        tps, conf = _find_gcps(image)

        # init list for all transformed points
        tps_transformed_list = []

        # transform points batchwise -> faster
        for j in range(0, tps.shape[0], batch_size):
            # Select a batch of points
            tps_batch = tps[j:j + batch_size, :]

            # Convert 2D points to 3D (homogeneous coordinates) for the batch
            tps_hom = np.hstack((tps_batch, np.ones((tps_batch.shape[0], 1))))

            # Apply the transformation matrix to the batch
            tps_transformed_homogeneous = tps_hom @ transform.T

            # Convert back to 2D points by ignoring the last component and store them
            tps_abs_batch = tps_transformed_homogeneous[:, :2].astype(int)
            tps_transformed_list.append(tps_abs_batch)

        # Concatenate all the processed batches
        tps_abs = np.vstack(tps_transformed_list)

        print(tps_abs.shape)

        # attach id and absolute coords to tps
        tps = np.concatenate((tps, tps_abs), axis=1)

        dict_gcps[i] = tps
        dict_conf[i] = conf

    overlapping_points = _identify_overlapping_gcps(dict_gcps, 10)

    return overlapping_points

def _find_gcps(image):
    sg_image = copy.deepcopy(image)
    sg_image = torch.from_numpy(sg_image)[None][None] / 255.

    # init device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load extractor and find points
    extractor = SuperPoint(max_num_keypoints=None).eval().to(device)
    features = extractor.extract(sg_image.to(device))

    key_points = features["keypoints"][0].cpu().detach().numpy().astype(int)
    scores = features["keypoint_scores"][0].cpu().detach().numpy()

    return key_points, scores

def _identify_overlapping_gcps(dict_tps, tolerance_abs):

    grouped_points = []

    def are_points_close(point1, point2, tolerance):
        return np.linalg.norm(np.array(point1) - np.array(point2)) <= tolerance

    total_points = sum(len(gcps) for gcps in dict_tps.values())

    # init progress bar
    progress_bar = tqdm(total=total_points)

    for id, gcps in dict_tps.items():
        for gcp in gcps:
            x_rel, y_rel, x_abs, y_abs = gcp
            point_added = False

            for group in grouped_points:
                if are_points_close(group['abs_coord'], (x_abs, y_abs), tolerance_abs):
                    group['occurrences'].append((id, x_rel, y_rel, x_abs, y_abs))
                    point_added = True
                    break

            # If point is not close to any group, create a new group
            if not point_added:
                grouped_points.append({
                    'abs_coord': (x_abs, y_abs),
                    'occurrences': [(id, x_rel, y_rel, x_abs, y_abs)]
                })

            # Update progress bar
            progress_bar.update(1)

    # print(IF AVERAGING POINTS; I ALSO NEED TO CHANGE THIS IN THE RELATIVE VALUES)

    print("Filter groups")

    # Filter out groups that don't meet the criteria (appear in at least 2 different images)
    common_points = {}
    for group in grouped_points:
        image_ids = set()
        relative_positions = []
        abs_coords_list = []  # To store the absolute coordinates for averaging

        for img_id, x_rel, y_rel, x_abs, y_abs in group['occurrences']:
            image_ids.add(img_id)
            relative_positions.append((x_rel, y_rel))
            abs_coords_list.append((x_abs, y_abs))  # Collect all absolute coordinates

        if len(image_ids) >= 2:
            abs_coords = np.array([occ[-2:] for occ in group['occurrences']])
            avg_abs_coord = np.mean(abs_coords, axis=0)
            key = tuple(avg_abs_coord)
            common_points[key] = {
                'image_ids': list(image_ids),
                'relative_positions': relative_positions,
                'avg_abs_coord': avg_abs_coord,  # Store the average absolute coordinates here
                'abs_coords': abs_coords_list  # Optionally, store the original absolute coordinates
            }

    return common_points

