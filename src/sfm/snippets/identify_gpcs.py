import copy
import numpy as np
import pandas as pd
import torch

from numba import jit, types
from numba.typed import List, Dict

from tqdm import tqdm
from typing import Tuple

from external.lightglue import SuperPoint


def identify_gcps(image_ids, images, transforms, masks=None, debug=False):
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

    # set variable for pandas dataframe
    data_gcps = None

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

        # mask the points if a mask is provided
        if masks is not None:
            # get the mask for the image
            mask = masks[i]

            # check if the shape of the mask is the same as the image
            if mask.shape != image.shape:
                raise ValueError(f"Shape of mask for image {image_id} does not match the image shape.")

            # Get the values of the mask at the points
            filter_vals = np.asarray([mask[int(row[1]), int(row[0])] for row in tps])

            # Determine indices to keep (logical OR to find any zeros, then invert)
            keep_indices = np.logical_not(filter_vals == 0)

            # Filter the tie points and confidences
            tps = tps[keep_indices]
            conf = conf[keep_indices]

        if np.any(tps < 0):
            raise ValueError("Negative x coordinate found in points for {image_id}.")

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

        # this list contains all data
        data_lst = []

        for idx, point in enumerate(tps):
            data_lst.append({
                "image_id": image_id,
                "x": point[0],
                "y": point[1],
                "x_abs": point[2],
                "y_abs": point[3],
                "conf": conf[idx]
            })

        # convert list of dicts to dataframe
        data_fr = pd.DataFrame(data_lst, columns=["image_id", "x", "y", "x_abs", "y_abs", "conf"])

        if data_gcps is None:
            data_gcps = data_fr
        else:
            # append to the dataframe
            data_gcps = pd.concat([data_gcps, data_fr], ignore_index=True)

    # get all rows with conf > 0.01
    data_gcps = data_gcps[data_gcps["conf"] > 0.01]

    overlapping_points = _identify_overlapping_gcps(data_gcps, 5)

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


def _identify_overlapping_gcps(data, threshold):
    from sklearn.cluster import DBSCAN

    if threshold == 0:
        threshold = 0.001

    # cluster the points with a given threshold
    clustering = DBSCAN(eps=threshold, min_samples=1, metric='manhattan')  # using manhattan distance
    data['cluster'] = clustering.fit_predict(data[['x_abs', 'y_abs']])

    # remove all clusters with only one point
    cluster_sizes = data['cluster'].value_counts()
    data = data[data['cluster'].map(cluster_sizes) > 1]

    # get the average abs coords for each cluster
    mean_coords = data.groupby('cluster')[['x_abs', 'y_abs']].mean()

    # get the average confidence for each cluster
    mean_conf = data.groupby('cluster')['conf'].mean()

    # Reset index to allow for merging
    mean_coords = mean_coords.reset_index()
    mean_conf = mean_conf.reset_index()

    # Merge the average coordinates and conf back into the original DataFrame
    data = data.merge(mean_coords, on='cluster', suffixes=('', '_avg'))
    data = data.merge(mean_conf, on='cluster', suffixes=('', '_avg'))

    # Replace x_abs and y_abs with their average values
    data['x_abs'] = data['x_abs_avg']
    data['y_abs'] = data['y_abs_avg']
    data['conf'] = data['conf_avg']

    # Drop the temporary average columns
    data.drop(columns=['x_abs_avg', 'y_abs_avg', 'conf_avg'], inplace=True)

    # Convert DataFrame to dictionary grouped by cluster
    cluster_dict = {}
    for cluster, group in data.groupby('cluster'):
        # Create a key for the cluster using the mean of x_abs and y_abs
        key = (group['x_abs'].mean(), group['y_abs'].mean())
        # Create a list of dictionaries for each row in the group
        cluster_dict[key] = group.drop(columns=['cluster', 'x_abs', 'y_abs']).to_dict('records')

    return cluster_dict
