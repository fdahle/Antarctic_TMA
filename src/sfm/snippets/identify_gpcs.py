import copy
import numpy as np
import torch

from tqdm import tqdm
from typing import Tuple

from external.lightglue import SuperPoint


def identify_gcps(image_ids, images, transforms, debug=False):
    """
    Identifies Ground Control Points (GCPs) across a list of images. Points are identified using the SuperPoint
    algorithm and converted to absolute coordinates using the provided transformation matrices. Overlapping points
    are then identified to obtain a list of GCPs that are present in multiple images.
    Args:
        image_ids (List[str]): A list of image IDs.
        images (List[np.ndarray]): A list of images in numpy array format.
        transforms (List[np.ndarray]): A list of transformation matrices corresponding to each image.
    Returns:
        Dict[Tuple[float, float], Dict[str, List]]: A dictionary where keys are tuples of averaged absolute coordinates
        of identified GCPs, and values are dictionaries containing image IDs, relative positions, the average absolute
        coordinates, and a list of original absolute coordinates of the GCPs.
    """

    # save gcps and confidence for each image
    dict_gcps = {}
    dict_conf = {}

    # iterating tps in batches is faster
    batch_size = 100

    # iterate all images
    for i, image_id in enumerate(image_ids):

        # get the image and transform for the image-id
        image = images[i]
        transform = transforms[i]

        # check if the transform is empty
        if np.all(transform == 0):
            continue

        if debug:
            print("Find interesting points for image: ", image_id)

        # find interesting point in the images
        tps, conf = _find_gcps(image)

        if debug:
            print(f"Found {tps.shape[0]} interesting points")

        # init list for all transformed points
        tps_transformed_list = []

        # transform points batch-wise -> faster
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

        if debug:
            print("Points are transformed")

        # Concatenate all the processed batches
        tps_abs = np.vstack(tps_transformed_list)

        # attach id and absolute coords to tps
        tps = np.concatenate((tps, tps_abs), axis=1)

        dict_gcps[image_id] = tps
        dict_conf[image_id] = conf

    # get overlapping points between images
    overlapping_points = _identify_overlapping_gcps(dict_gcps, 10, debug)

    return overlapping_points


def _find_gcps(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds keypoints and their scores in a given image using the SuperPoint extractor.
    Args:
        image (np.ndarray): The image array in which to find keypoints.
    Returns:
        keypoints (np.ndarray): numpy array of keypoint coordinates with (x, y).
        scores (np.ndarray): numpy array of corresponding scores for each keypoint.
    """

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


def _identify_overlapping_gcps(dict_tps: dict[int, np.ndarray], tolerance_abs: float, debug) -> \
        dict[Tuple[float, float], dict[str, list]]:
    """
    Identify and groups ground control points (GCPs) across images based on the proximity of their absolute
    coordinates. Each point is compared against existing groups of points. If a point is within a specified tolerance
    of a group absolute coordinates, it is added to that group. Otherwise, a new group is formed. Averages the absolute
    coordinates of grouped points and filters out any groups that do not appear in at least two different images.

    Args:
        dict_tps: A dictionary mapping image IDs to arrays of ground control points. Each GCP is represented as a
            tuple of (x_rel, y_rel, x_abs, y_abs), where `x_rel` and `y_rel` are the relative coordinates in the
            image, and `x_abs` and `y_abs` are the absolute coordinates in some reference frame.
        tolerance_abs: A float specifying the tolerance within which two points are considered to be at
            the same location in terms of their absolute coordinates.

    Returns:
        A dictionary where each key is a tuple representing the averaged absolute coordinates of a group of points,
            and each value is a dictionary containing:
            - 'image_ids': A list of image IDs in which the grouped points appear.
            - 'relative_positions': A list of tuples representing the relative positions of each point in the group
                within their respective images.
            - 'avg_abs_coord': A tuple representing the averaged absolute coordinates of the points in the group.
            - 'abs_coords': A list of tuples representing the original absolute coordinates of all points in the group.
    """

    if debug:
        print("Identify overlapping points")

    # store all grouped points in this list
    grouped_points = []

    # Calculate the total number of points to process
    total_points = sum(len(gcps) for gcps in dict_tps.values())

    # init progress bar
    progress_bar = tqdm(total=total_points)

    # iterate all images
    for img_id, gcps in dict_tps.items():

        # iterate all gcps of one image
        for gcp in gcps:

            # reset point_added
            point_added = False

            # get relative and absolute coords
            x_rel, y_rel, x_abs, y_abs = gcp

            # check already added points
            for group in grouped_points:

                # get distance between point
                distance = np.linalg.norm(np.array(group['abs_coord']) - np.array((x_abs, y_abs)))

                # check if point is close to a group
                if distance <= tolerance_abs:
                    group['abs_coord'].append((x_abs, y_abs))
                    group['rel_coord'].append((x_rel, y_rel))
                    group['img_id'].append(img_id)
                    point_added = True
                    break

            # If point is not close to any group, create a new group
            if not point_added:
                grouped_points.append({
                    'abs_coord': [(x_abs, y_abs)],
                    'rel_coord': [(x_rel, y_rel)],
                    'img_id': [img_id]
                })

            # Update progress bar
            progress_bar.update(1)

    # TODO (IF AVERAGING POINTS; I ALSO NEED TO CHANGE THIS IN THE RELATIVE VALUES)

    # Filter out groups that don't meet the criteria (appear in at least 2 different images)
    common_points = {}

    # iterate all groups
    for group in grouped_points:

        # Use a set to ensure that unique image IDs are counted
        image_ids = set(group['img_id'])

        # only add points that are in at least 2 images
        if len(image_ids) >= 2:

            # get abs and rel coords
            abs_coords = np.array(group['abs_coord'])
            rel_coords = group['rel_coord']

            # Average the absolute coordinates since we're considering them for common points
            avg_abs_coord = np.mean(abs_coords, axis=0)
            key = tuple(avg_abs_coord)

            # Prepare the absolute coordinates for output
            abs_coords_list = group['abs_coord']  # This already contains all absolute coordinates

            # Store the averaged absolute coordinates and other info in common_points
            common_points[key] = {
                'image_ids': list(image_ids),
                'rel_coords': rel_coords,
                'avg_abs_coord': avg_abs_coord,
                'abs_coords': abs_coords_list
            }

    return common_points
