import numpy as np
import os

from tqdm import tqdm

import src.base.find_tie_points as ftp
import src.load.load_image as li

def find_tie_points_for_sfm(image_folder, combinations,
                            mask_folder=None,
                            rotation_dict=None,
                            max_degree_diff=45,
                            min_tps=0, max_tps=None,
                            min_tp_confidence=0,
                            tp_type=float):

    # init the tie point detector
    tpd = ftp.TiePointDetector('lightglue',
                               min_conf=min_tp_confidence, tp_type=tp_type)

    # dict to save values
    tp_dict = {}
    conf_dict = {}

    # keep track of the last image
    old_img_1_id = None

    # get all unique ids from the combinations
    all_ids = {k for k in combinations} | {v for values in combinations.values() for v in values}

    # check rotation dict for completeness
    if rotation_dict is not None:
        missing_ids = [img_id for img_id in all_ids if img_id not in rotation_dict]
        if len(missing_ids) > 0:
            raise ValueError(f"Missing rotation data for {len(missing_ids)} images: {missing_ids}")
    else:
        # create a rotation dict with 0 degrees for all images
        rotation_dict = {img_id: 0 for img_id in all_ids}


    # loop over the combinations
    for idx, (img_1_id, img_2_id) in (pbar := tqdm(enumerate(combinations),
                                                   total=len(combinations))):

        pbar.set_description(f"Finding tie points for {img_1_id} and {img_2_id}")

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

        # check the difference in rotations
        if "V" in img_1_id and "V" in img_2_id:

            # get difference in rotation
            rot_1 = rotation_dict[img_1_id]
            rot_2 = rotation_dict[img_2_id]
            rot_diff = rot_2 - rot_1

            # if the difference is larger than threshold rotate the images
            if abs(rot_diff) > max_degree_diff:
                img_1_rotated, rot_mat_1 = ri.rotate_image(img_1, rot_1,
                                                           return_rot_matrix=True)
                img_2_rotated, rot_mat_2 = ri.rotate_image(img_2, rot_2,
                                                           return_rot_matrix=True)
                if mask_1 is not None:
                    mask_1_rotated = ri.rotate_image(mask_1, rot_1,
                                                     return_rot_matrix=True)
                else:
                    mask_1_rotated = None
                if mask_2 is not None:
                    mask_2_rotated = ri.rotate_image(mask_2, rot_2,
                                                     return_rot_matrix=True)
                else:
                    mask_2_rotated = None

            else:
                img_1_rotated = img_1
                img_2_rotated = img_2
                mask_1_rotated = mask_1
                mask_2_rotated = mask_2
                rot_mat_1 = np.eye(3)
                rot_mat_2 = np.eye(3)

        # find tie points
        tps, conf = tpd.find_tie_points(img_1_rotated, img_2_rotated,
                                        mask1=mask_1_rotated, mask2=mask_2_rotated)

        tps[:, :2] = rp.rotate_points(tps[:, :2], rot_mat_1, invert=True)
        tps[:, 2:] = rp.rotate_points(tps[:, 2:], rot_mat_2, invert=True)

        # rotate img_1 180 degrees if we lack enough tps
        if tps.shape[0] < min_tps:
            img_1_rotated = ri.rotate_image(img_1, 180)


        if np.amax(tps[:,0]) > img_1_dims[1] or np.amax(tps[:,1]) > img_1_dims[0]:
            raise ValueError(f"Tie points for {img_1_id} exceed image dimensions")
        if np.amax(tps[:,2]) > img_2_dims[1] or np.amax(tps[:,3]) > img_2_dims[0]:
            raise ValueError(f"Tie points for {img_2_id} exceed image dimensions")

        # limit number of tie points
        if max_tps is not None:

            # select the top points and conf
            top_indices = np.argsort(conf)[-max_tps:][::-1]
            tps = tps[top_indices]
            conf = conf[top_indices]

        # save the tie points and conf
        tp_dict[(img_1_id, img_2_id)] = tps
        conf_dict[(img_1_id, img_2_id)] = conf

    return tp_dict, conf_dict