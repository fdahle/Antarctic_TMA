"""Geo-reference a relative orthomosaic"""

import numpy as np

import src.base.calc_bounds as cb
import src.base.find_tie_points as ftp
import src.base.rotate_image as ri
import src.base.rotate_points as rp

import src.georef.snippets.calc_transform as ct

def georef_ortho(relative_ortho: np.ndarray,
                 absolute_ortho: np.ndarray,
                 absolute_transform: affine.Transform,
                 auto_rotate: bool = False,
                 rotation_step: int = 10,
                 min_nr_tps: int = 0,
                 min_conf: float = 0.0,
                 start_conf: float = 0.9,
                 tp_type: type =float):

    # init tie point detector
    tpd = ftp.TiePointDetector('lightglue',
                               min_conf=min_conf,
                               tp_type=tp_type)

    if auto_rotate:
        rotation_values = list(range(0, 360, rotation_step))
    else:
        rotation_values = [0]

    best_tps = np.empty((0, 4))
    best_conf = np.empty((0, 1))
    best_rotation = 0
    best_rot_mat = None

    # iterate over all rotation values
    for rotation in rotation_values:

        # rotate the relative ortho
        rotated_ortho, rot_mat = ri.rotate_image(relative_ortho, rotation,
                                                 return_rot_matrix=True)

        # find tie points
        tps, conf = tpd.find_tie_points(rotated_ortho, absolute_ortho)

        if tps.shape[0] > best_tps.shape[0]:
            best_tps = tps
            best_conf = conf
            best_rotation = rotation
            best_rot_mat = rot_mat

    # get the best values
    all_tps = best_tps
    all_conf = best_conf
    rotation = best_rotation
    rot_mat = best_rot_mat

    # set start confidence
    conf_val = start_conf

    # init tie points
    tps = np.empty((0, 4))

    # filter tie points based on confidence
    while conf_val >= min_conf:
        tps = all_tps[all_conf >= conf_val]
        conf = all_conf[all_conf >= conf_val]

        if tps.shape[0] >= min_nr_tps:
            break

        conf_val -= 0.05

    if tps.shape[0] < min_nr_tps:
        raise ValueError(f"Not enough tie points found: {tps.shape[0]} < {min_nr_tps}")

    print("Found {} tie points with a confidence of min {}".format(
        tps.shape[0], conf_val))

    # rotate points back
    tps[:, 2:] = rp.rotate_points(tps[:, 2:],
                                  rot_mat, invert=True)

    # convert points to absolute values
    absolute_points = np.array([absolute_transform * tuple(point) for point in tps[:, 0:2]])
    tps[:, 0:2] = absolute_points

    transform, residuals = ct.calc_transform(relative_ortho, tps,
                                             transform_method="rasterio",
                                             gdal_order=3)

    # get bounds from the transform
    bounds = cb.calc_bounds(transform, relative_ortho.shape)

    return transform, bounds, rotation