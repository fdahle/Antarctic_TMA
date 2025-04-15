import numpy as np
import os

from tqdm import tqdm

import src.base.find_tie_points as ftp
import src.base.rotate_image as ri
import src.base.rotate_points as rp
import src.display.display_images as di
import src.load.load_image as li

from src.sfm_agi2.SfMError import SfMError

def find_tie_points_for_sfm(image_folder: str,
                            combinations: dict[str, list[str]],
                            mask_folder: str | None = None,
                            rotation_dict: dict[str, tuple[int, int, int]] = None,
                            max_degree_diff: int = 45,
                            min_tps: int = 0, max_tps: int | None =None,
                            min_tp_confidence=0,
                            rotate_180: bool = False,
                            tp_type: type = float,
                            debug: bool = False,
                            debug_folder: str | None = None,
                            use_cached_tps: bool = False,
                            save_cached_tps: bool = False,
                            cache_folder: str | None = None,) -> \
        tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:


    # check some params
    if min_tps < 0:
        raise ValueError("min_tps must be >= 0")
    if min_tp_confidence < 0 or min_tp_confidence > 1:
        raise ValueError("min_tp_confidence must be between 0 and 1")
    if max_tps is not None and max_tps < min_tps:
        raise ValueError("max_tps must be > min_tps")
    if tp_type != float and tp_type != int:
        raise ValueError("tp_type must be float or int")

    # create debug folder if needed
    if debug and debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    # init the tie point detector
    tpd = ftp.TiePointDetector('lightglue',
                               min_conf=min_tp_confidence, tp_type=tp_type)

    # dict to save values
    tp_dict = {}
    conf_dict = {}

    # get all unique ids from the combinations
    all_ids = {k for k in combinations} | {v for values in combinations.values() for v in values}

    # check rotation dict for completeness
    if rotation_dict is not None:
        missing_ids = [img_id for img_id in all_ids if img_id not in rotation_dict]
        if len(missing_ids) > 0:
            raise ValueError(f"Missing rotation data for {len(missing_ids)} images: {missing_ids}")
    else:
        # create a rotation dict with 0 degrees for all images
        rotation_dict = {img_id: (0, 0 , 0) for img_id in all_ids}

    # keep track of the last image
    old_img_1_id = None

    # init already some variables
    img_1 = None
    img_1_dims = None
    mask_1 = None
    img_2 = None
    tps = None
    conf = None

    # get flattened combinations and their number
    all_pairs = ((k, v) for k, vs in combinations.items() for v in vs)
    total_combinations = sum(len(v) for v in combinations.values())

    # create progress bar
    pbar = tqdm(total=total_combinations, desc="Find matches",
                position=0, leave=True)

    # loop over the combinations
    for img_1_id, img_2_id in all_pairs:

        # update the progress bar description
        pbar.set_postfix_str(f"{img_1_id} - {img_2_id}")

        # flag if cached tps are loaded
        cached_tps_loaded = False

        # check if we can use cached tps
        if use_cached_tps:

            # get cached tps
            tps, conf = _load_cached_tps(cache_folder, img_1_id, img_2_id)

            # successfully loaded tps -> continue
            if tps is not None:
                cached_tps_loaded = True

        if cached_tps_loaded is False:
            # check if we need to load a new image 1
            if img_1_id != old_img_1_id:

                # load image 1
                path_img_1 = os.path.join(image_folder, img_1_id + ".tif")
                img_1 = li.load_image(path_img_1)
                img_1_dims = img_1.shape

                # load mask 1
                if mask_folder is not None:
                    path_mask_1 = os.path.join(mask_folder, img_1_id + ".tif")
                    mask_1 = li.load_image(path_mask_1)
                else:
                    mask_1 = None

                # set the old image id
                old_img_1_id = img_1_id

            # load image 2
            path_img_2 = os.path.join(image_folder, img_2_id + ".tif")
            img_2 = li.load_image(path_img_2)
            img_2_dims = img_2.shape

            # load mask 2
            if mask_folder is not None:
                path_mask_2 = os.path.join(mask_folder, img_2_id + ".tif")
                mask_2 = li.load_image(path_mask_2)
            else:
                mask_2 = None

            # default is no rotation
            img_1_rotated = img_1.copy()
            img_2_rotated = img_2.copy()
            mask_1_rotated = mask_1.copy() if mask_1 is not None else None
            mask_2_rotated = mask_2.copy() if mask_2 is not None else None
            rot_mat_1 = np.eye(3)
            rot_mat_2 = np.eye(3)

            # check the difference in rotations
            if "V" in img_1_id and "V" in img_2_id:

                # get difference in rotation
                rot_1 = rotation_dict[img_1_id][0]
                rot_2 = rotation_dict[img_2_id][0]
                rot_diff = rot_2 - rot_1

                # if the difference is larger than threshold rotate the images
                if abs(rot_diff) > max_degree_diff:
                    img_1_rotated, rot_mat_1 = ri.rotate_image(img_1, rot_1,
                                                               return_rot_matrix=True)
                    img_2_rotated, rot_mat_2 = ri.rotate_image(img_2, rot_2,
                                                               return_rot_matrix=True)
                    mask_1_rotated = ri.rotate_image(mask_1, rot_1) if mask_1 is not None else None
                    mask_2_rotated = ri.rotate_image(mask_2, rot_2) if mask_2 is not None else None

            # find tie points
            tps, conf = tpd.find_tie_points(img_1_rotated, img_2_rotated,
                                            mask1=mask_1_rotated, mask2=mask_2_rotated)

            # second check if we need more tps
            if tps.shape[0] < min_tps and rotate_180:

                # first try with image 1
                img_1_180, rot_mat_1_180 = ri.rotate_image(img_1_rotated, 180,
                                            return_rot_matrix=True)
                mask_1_180 = ri.rotate_image(mask_1_rotated, 180) if mask_1_rotated is not None else None

                tps_1, conf_1 = tpd.find_tie_points(img_1_180, img_2_rotated,
                                                    mask1=mask_1_180, mask2=mask_2_rotated)

                # second try with image 2
                img_2_180, rot_mat_2_180 = ri.rotate_image(img_2_rotated, 180,
                                            return_rot_matrix=True)
                mask_2_180 = ri.rotate_image(mask_2_rotated, 180) if mask_2_rotated is not None else None

                tps_2, conf_2 = tpd.find_tie_points(img_1_rotated, img_2_180,
                                                    mask1=mask_1_rotated, mask2=mask_2_180)

                # check if we have more tps
                rotated_image = None
                if tps_1.shape[0] > tps.shape[0]:
                    tps = tps_1
                    conf = conf_1
                    rotated_image = 1
                if tps_2.shape[0] > tps.shape[0]:
                    tps = tps_2
                    conf = conf_2
                    rotated_image = 2

                if rotated_image == 1:
                    tps[:, :2] = rp.rotate_points(tps[:, :2], rot_mat_1_180, invert=True)
                elif rotated_image == 2:
                    tps[:, 2:] = rp.rotate_points(tps[:, 2:], rot_mat_2_180, invert=True)

            # rotate the tie points back
            if tps.shape[0] > 0:
                tps[:, :2] = rp.rotate_points(tps[:, :2], rot_mat_1, invert=True)
                tps[:, 2:] = rp.rotate_points(tps[:, 2:], rot_mat_2, invert=True)

            if np.any(tps[:, 0] >= img_1_dims[1]) or np.any(tps[:, 1] >= img_1_dims[0]):
                raise ValueError(f"Tie points for {img_1_id} exceed image dimensions")
            if np.any(tps[:, 2] >= img_2_dims[1]) or np.any(tps[:, 3] >= img_2_dims[0]):
                raise ValueError(f"Tie points for {img_2_id} exceed image dimensions")

        # check if we have enough tps
        if tps.shape[0] < min_tps:
            # update the progress bar
            pbar.update(1)
            continue

        if debug:

            # load images for debugging if not loaded already
            path_img_1 = os.path.join(image_folder, img_1_id + ".tif")
            img_1 = li.load_image(path_img_1)
            path_img_2 = os.path.join(image_folder, img_2_id + ".tif")
            img_2 = li.load_image(path_img_2)

            dbg_file_name = f"{img_1_id}_{img_2_id}.png"
            dbg_img_path = os.path.join(debug_folder, dbg_file_name)
            style_config={
                "title": f"{tps.shape[0]} tps for {img_1_id} - {img_2_id}",
            }
            di.display_images([img_1, img_2],
                              tie_points=tps, tie_points_conf=conf,
                              style_config=style_config,
                              save_path=dbg_img_path)

        # save tie points to cache
        if save_cached_tps and cached_tps_loaded is False:
            _save_tps_to_cache(cache_folder, img_1_id, img_2_id, tps, conf)

        # limit number of tie points
        if max_tps is not None:

            # select the top points and conf
            top_indices = np.argsort(conf)[-max_tps:][::-1]
            tps = tps[top_indices]
            conf = conf[top_indices]

        # save the tie points and conf
        tp_dict[(img_1_id, img_2_id)] = tps
        conf_dict[(img_1_id, img_2_id)] = conf

        # update the progress bar
        pbar.update(1)

    # close the progress bar
    pbar.set_postfix_str("- Finished -")
    pbar.close()

    # raise error if no tie points found
    if len(tp_dict) == 0:
        raise SfMError("No tie points found for any image pair.")

    return tp_dict, conf_dict

def _load_cached_tps(tps_fld, img_1_id, img_2_id):
    for f_name in [f"{img_1_id}_{img_2_id}.txt", f"{img_2_id}_{img_1_id}.txt"]:
        file_path = os.path.join(tps_fld, f_name)
        if not os.path.isfile(file_path):
            continue

        if os.stat(file_path).st_size > 0:
            data = np.atleast_2d(np.loadtxt(file_path))
            if f_name.startswith(f"{img_2_id}_{img_1_id}"):
                tps = data[:, [2, 3, 0, 1]]  # reorder to x1, y1, x2, y2
                conf = data[:, 4]
            else:
                tps, conf = data[:, :4], data[:, 4]
        else:
            tps, conf = np.zeros((0, 4)), np.zeros(0)

        return tps, conf

    return None, None

def _save_tps_to_cache(tps_fld, img_1_id, img_2_id, tps, conf):
    if img_1_id < img_2_id:
        file_path = os.path.join(tps_fld, f"{img_1_id}_{img_2_id}.txt")
        np.savetxt(file_path, np.hstack((tps, conf[:, None])),
                   fmt='%f')
    else:
        file_path = os.path.join(tps_fld, f"{img_2_id}_{img_1_id}.txt")
        tps_switched = np.hstack((tps[:, 2:], tps[:, :2]))
        np.savetxt(file_path, np.hstack((tps_switched, conf[:, None])),
                   fmt='%f')
