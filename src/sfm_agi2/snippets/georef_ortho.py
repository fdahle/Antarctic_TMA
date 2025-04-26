"""Geo-reference a relative orthomosaic"""

import numpy as np
import os

from rasterio.transform import Affine
from tqdm import tqdm

import src.georef.snippets.apply_transform as at
import src.base.calc_bounds as cb
import src.base.find_tie_points as ftp
import src.base.resize_image as rei
import src.base.rotate_image as ri
import src.base.rotate_points as rp
import src.display.display_images as di
import src.georef.snippets.calc_transform as ct
from src.sfm_agi2.SfMError import SfMError

debug_show_rotation = False
debug_min_tps = 25

def georef_ortho(relative_ortho: np.ndarray,
                 absolute_ortho: np.ndarray,
                 absolute_transform: Affine,
                 input_rotation: int | None = None,
                 auto_rotate: bool = False,
                 rotation_step: int = 10,
                 min_nr_tps: int = 0,
                 min_conf: float = 0.0,
                 start_conf: float = 0.9,
                 trim_image: bool = True,
                 trim_threshold: float = 0.75,
                 tp_type: type =float,
                 debug: bool = False,
                 debug_folder: str | None = None,
                 data_folder: str | None = None
                 ):

    # check shape of ortho abs
    if absolute_ortho.ndim == 2:
        absolute_ortho = absolute_ortho[np.newaxis, :, :]
    elif absolute_ortho.ndim == 3:
        if absolute_ortho.shape[0] > 10:
            raise ValueError("Bands for the modern ortho should be in the "
                             "first dimension!")

    # create debug folder if needed
    if debug and debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    # create data folder if needed
    if data_folder is not None:
        os.makedirs(data_folder, exist_ok=True)

    # get 2d shape of ortho abs
    absolute_ortho_shape = (absolute_ortho.shape[1], absolute_ortho.shape[2])

    # init tie point detector
    tpd = ftp.TiePointDetector('lightglue',
                               min_conf=min_conf,
                               tp_type=tp_type)

    # already init trim vars
    trimmed_ortho = relative_ortho
    trim_x_min = 0
    trim_y_min = 0

    # trim the image if needed
    if trim_image:
        # Calculate the fraction of white pixels for each row and column
        row_white_fraction = np.mean(relative_ortho == 255, axis=1)
        col_white_fraction = np.mean(relative_ortho == 255, axis=0)

        # Identify rows and columns that are not mostly white
        rows_to_keep = np.where(row_white_fraction < trim_threshold)[0]
        cols_to_keep = np.where(col_white_fraction < trim_threshold)[0]

        if rows_to_keep.size and cols_to_keep.size:
            # Determine the bounding box for rows and columns to keep
            trim_y_min = rows_to_keep[0]
            trim_y_max = rows_to_keep[-1] + 1  # +1 to include the last index
            trim_x_min = cols_to_keep[0]
            trim_x_max = cols_to_keep[-1] + 1

            # Trim the image using the calculated indices
            trimmed_ortho = relative_ortho[trim_y_min:trim_y_max,
                            trim_x_min:trim_x_max]

    # resize the ortho to the absolute ortho size
    relative_ortho_resized, s_x, s_y = rei.resize_image(trimmed_ortho,
                                      absolute_ortho_shape,
                                      return_scale_factor=True)  # noqa

    # get the rotation values
    if input_rotation is not None:
        rotation_values = [input_rotation]
    elif auto_rotate:
        rotation_values = list(range(0, 360, rotation_step))
    else:
        rotation_values = [0]

    best_tps = np.empty((0, 4))
    best_conf = np.empty((0, 1))
    best_rotation = 0
    best_rot_mat = None

    # create progress bar
    pbar = tqdm(total=len(rotation_values), desc="Find matches for georef",
                position=0, leave=True)

    # iterate over all rotation values
    for rotation in rotation_values:

        # update progress bar description
        pbar.set_postfix_str(f"Rotation: {rotation}° - "
                             f"best: {best_tps.shape[0]} tie points at {best_rotation}°")

        # rotate the relative ortho
        rotated_ortho, rot_mat = ri.rotate_image(relative_ortho_resized, rotation,
                                                 return_rot_matrix=True)

        # find tie points
        tps: np.ndarray
        tps, conf = tpd.find_tie_points(absolute_ortho, rotated_ortho)

        if debug_show_rotation:
            if tps.shape[0] < debug_min_tps:
                continue
            style_config = {
                'title': f"Rotation: {rotation}°: {tps.shape[0]} tie points",
            }
            di.display_images([absolute_ortho, rotated_ortho],
                              tie_points=tps, tie_points_conf=conf,
                              style_config=style_config)

        if tps.shape[0] > best_tps.shape[0]:
            best_tps = tps
            best_conf = conf
            best_rotation = rotation
            best_rot_mat = rot_mat

        pbar.update(1)

    # close progress bar
    pbar.set_postfix_str("- Finished -")
    pbar.close()

    # get the best values
    all_tps = best_tps
    all_conf = best_conf
    rotation = best_rotation
    rot_mat = best_rot_mat

    # set start confidence
    conf_val = start_conf

    # init tie points
    tps = np.empty((0, 4))
    conf = np.empty((0, 1))

    # filter tie points based on confidence
    while conf_val >= min_conf:
        tps = all_tps[all_conf >= conf_val]
        conf = all_conf[all_conf >= conf_val]

        if tps.shape[0] >= min_nr_tps:
            break

        conf_val -= 0.05

    # rotate points back
    tps[:, 2:] = rp.rotate_points(tps[:, 2:],
                                  rot_mat, invert=True)

    # scale points back
    tps[:, 2] = tps[:, 2] / s_x
    tps[:, 3] = tps[:, 3] / s_y

    # account for the trimming
    tps[:, 2] = tps[:, 2] + trim_x_min
    tps[:, 3] = tps[:, 3] + trim_y_min


    if debug:
        dbg_file_name = "georef_ortho.png"
        dbg_img_path = os.path.join(debug_folder, dbg_file_name)
        di.display_images([absolute_ortho, relative_ortho],
                          tie_points=tps, tie_points_conf=conf,
                          save_path=dbg_img_path)

    if tps.shape[0] < min_nr_tps:
        raise SfMError(f"Not enough tie points found: {tps.shape[0]} < {min_nr_tps}")

    print("[INFO] Found {} tie points with a confidence of min {} at rotation".format(
        tps.shape[0], round(conf_val, 2), rotation))

    # convert points to absolute values
    absolute_points = np.array([absolute_transform * tuple(point) for point in tps[:, 0:2]])
    tps[:, 0:2] = absolute_points

    transform, residuals = ct.calc_transform(relative_ortho, tps,
                                             transform_method="rasterio",
                                             gdal_order=3)

    if data_folder is not None:

        # save the geo-referenced ortho
        georef_data_pth = os.path.join(data_folder, "ortho_georef.tif")
        at.apply_transform(relative_ortho, transform,
                           georef_data_pth)

        # save the tie points
        conf = conf.reshape(-1, 1)  # Converts shape from (N,) to (N, 1)
        tps_conf = np.hstack((tps, conf))
        tps_data_path = os.path.join(data_folder, "tie_points.csv")
        col_names = "x_abs; y_abs; x_rel; y_rel; conf"
        np.savetxt(tps_data_path, tps_conf, delimiter=";", header=col_names)

    # get bounds from the transform
    bounds = cb.calc_bounds(transform, relative_ortho.shape)

    return transform, bounds, rotation