import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

import src.base.find_overlapping_images as foi
import src.base.find_tie_points as ftp
import src.display.display_images as di
import src.load.load_image as li

import src.base.rotate_image as ri
import src.base.rotate_points as rp

debug_show_intermediate_steps = False
debug_show_final_tps = False
debug_min_tps = 0

TPS_TXT_FLD = "/data/ATM/data_1/sfm/agi_data/tie_points"

def find_tie_points_for_sfm(img_folder, mask_folder=None,
                            footprint_dict=None,
                            rotation_dict=None,
                            matching_method='all',
                            tp_type=float,
                            min_overlap=0.5,
                            min_conf=0.7,
                            min_tps=25, max_tps=4000,
                            use_cached_tps=False,
                            save_tps=False):

    # init list and dicts for later usage
    image_ids = []  # list with image ids
    combinations = []  # list with image combinations
    image_dict = {}  # dict with the loaded images
    mask_dict = {}  # dict with the loaded masks

    # get a list with all already calculated tie points
    if use_cached_tps:
        folder_path = Path(TPS_TXT_FLD)
        cached_tps = [file.stem for file in folder_path.glob("*.txt")]
    else:
        cached_tps = []

    # get the tif-images in the folder
    for filename in os.listdir(img_folder):
        if filename.lower().endswith('.tif'):
            image_ids.append(filename[:-4])

    # create a list with all combinations of images
    if matching_method == 'all':

        # create a list with all combinations of images
        for i in range(len(image_ids)):
            for j in range(i+1, len(image_ids)):

                # skip identical images
                if image_ids[i] == image_ids[j]:
                    continue

                if (image_ids[j], image_ids[i]) in combinations:
                    continue

                # add the combination to the list
                combinations.append((image_ids[i], image_ids[j]))
    elif matching_method == 'sequential':
        # create a list with all sequential combinations of images
        for i in range(len(image_ids)-1):

            # add the combination to the list
            combinations.append((image_ids[i], image_ids[i+1]))
    elif matching_method == 'overlap':

        # get the footprints as lst
        footprints_lst = [footprint_dict[image_id] for image_id in image_ids]

        # get the overlapping images
        overlap_dict = foi.find_overlapping_images(image_ids, footprints_lst,
                                                   min_overlap=min_overlap)

        # create a list with all combinations of images
        for img_id, overlap_lst in overlap_dict.items():

            # iterate over all overlapping images
            for overlap_id in overlap_lst:

                # skip identical images
                if img_id == overlap_id:
                    continue

                # check if the combination is already in the list
                if (overlap_id, img_id) in combinations:
                    continue

                # add the combination to the list
                combinations.append((img_id, overlap_id))
    else:
        raise NotImplementedError(f"Matching method {matching_method} not existing")

    # init tie-point detector
    tpd = ftp.TiePointDetector('lightglue', verbose=True,
                               min_conf_value=min_conf, tp_type=tp_type,
                               display=debug_show_intermediate_steps)

    # dict for the tie points & confidence values
    tp_dict = {}
    conf_dict = {}

    # iterate over all combinations
    print("  Iterate combinations")
    for img_1_id, img_2_id in (pbar := tqdm(combinations)):

        # check if the tie points are already calculated
        if use_cached_tps and img_1_id + "_" + img_2_id in cached_tps:

            # define the file path
            file_path = TPS_TXT_FLD + "/" + img_1_id + "_" + img_2_id + ".txt"

            # check file size of the cached tie points
            file_size = os.stat(file_path).st_size

            # only load the file if it is not empty
            if file_size > 0:

                data = np.loadtxt(file_path)

                # assure data is 2D
                data = np.atleast_2d(data)

                tps = data[:, :4]
                conf = data[:, 4]
            else:
                tps = np.zeros((0, 4))
                conf = np.zeros((0))

            print(f"{tps.shape[0]} tie points between {img_1_id} and {img_2_id} (cached)")

        elif use_cached_tps and img_2_id + "_" + img_1_id in cached_tps:

            # define the file path
            file_path = TPS_TXT_FLD + "/" + img_2_id + "_" + img_1_id + ".txt"

            # check file size of the cached tie points
            file_size = os.stat(file_path).st_size

            # only load the file if it is not empty
            if file_size > 0:

                data = np.loadtxt(file_path)

                # assure data is 2D
                data = np.atleast_2d(data)

                tps = data[:, :4]
                conf = data[:, 4]

                # switch the first two and last two columns of the tie points
                tps = np.hstack((tps[:, 2:], tps[:, :2]))
            else:
                tps = np.zeros((0, 4))
                conf = np.zeros((0))

            print(f"{tps.shape[0]} tie points between {img_1_id} and {img_2_id} (cached)")

        # we need to calculate the tie points
        else:
            # load image 1
            if img_1_id not in image_dict:
                path_img1 = os.path.join(img_folder, img_1_id + ".tif")
                image_dict[img_1_id] = li.load_image(path_img1)
            image1 = image_dict[img_1_id]

            # load image 2
            if img_2_id not in image_dict:
                path_img2 = os.path.join(img_folder, img_2_id + ".tif")
                image_dict[img_2_id] = li.load_image(path_img2)
            image2 = image_dict[img_2_id]

            # load masks
            if mask_folder is not None:

                # load mask 1
                if img_1_id not in mask_dict:
                    path_mask1 = os.path.join(mask_folder, img_1_id + ".tif")
                    path_mask1 = path_mask1.replace('.tif', '_mask.tif')
                    mask_dict[img_1_id] = li.load_image(path_mask1)
                mask1 = mask_dict[img_1_id]

                # load mask 2
                if img_2_id not in mask_dict:
                    path_mask2 = os.path.join(mask_folder, img_2_id + ".tif")
                    path_mask2 = path_mask2.replace('.tif', '_mask.tif')
                    mask_dict[img_2_id] = li.load_image(path_mask2)
                mask2 = mask_dict[img_2_id]
            else:

                # no masks available
                mask1 = None
                mask2 = None

            # init rotmat variable
            rotmat1 = None
            rotmat2 = None

            # rotate the images if necessary
            if rotation_dict is not None:

                # get the rotation angles
                rot1 = rotation_dict[img_1_id][0]
                rot2 = rotation_dict[img_2_id][0]

                rot_dif = abs(rot1 - rot2)

                # rotation is only required if both angles are very different
                if rot_dif > 22.5:

                    # rotate the images
                    image1, rotmat1 = ri.rotate_image(image1, rot1, return_rot_matrix=True)
                    image2, rotmat2 = ri.rotate_image(image2, rot2, return_rot_matrix=True)

                    # rotate the masks
                    if mask1 is not None:
                        mask1 = ri.rotate_image(mask1, rot1)
                    if mask2 is not None:
                        mask2 = ri.rotate_image(mask2, rot2)
            else:
                rot1 = 0
                rot2 = 0

            # get the tie points
            tps, conf = tpd.find_tie_points(image1, image2,
                                            mask1=mask1, mask2=mask2)

            # rotate the tie points back if necessary
            if tps.shape[0] > 0 and rotation_dict is not None:

                rot1 = rotation_dict[img_1_id][0]
                rot2 = rotation_dict[img_2_id][0]
                rot_dif = abs(rot1 - rot2)

                # rotation is only required if both angles are very different
                if rot_dif > 22.5:

                    tps[:, :2] = rp.rotate_points(tps[:, :2], rotmat1, invert=True)
                    tps[:, 2:] = rp.rotate_points(tps[:, 2:], rotmat2, invert=True)

            # check if there are negative values
            if np.any(tps < 0):
                raise ValueError("Negative tie points found")

            # make empty arrays 2dimension
            if tps.shape[0] == 0:
                tps = np.zeros((0, 4))
                conf = np.zeros((0))

            # check if tps are out of bounds
            if np.any(tps[:, 0] >= image1.shape[1]) or np.any(tps[:, 1] >= image1.shape[0]):
                raise ValueError("Tie points out of bounds")

            if save_tps:

                if img_1_id < img_2_id:
                    np.savetxt(TPS_TXT_FLD + "/" + img_1_id + "_" + img_2_id + ".txt",
                               np.hstack((tps, conf[:, None])), fmt='%f')
                else:
                    tps_switched = np.hstack((tps[:, 2:], tps[:, :2]))
                    np.savetxt(TPS_TXT_FLD + "/" + img_2_id + "_" + img_1_id + ".txt",
                                 np.hstack((tps_switched, conf[:, None])), fmt='%f')

            if debug_show_final_tps:
                style_config = {
                    'title': f"{tps.shape[0]} tie points between {img_1_id} and {img_2_id}",
                }

                di.display_images([image1, image2],
                                  tie_points=tps, tie_points_conf=conf,
                                  style_config=style_config)

            print(f"{tps.shape[0]} tie points between {img_1_id} ({rot1}) and {img_2_id} ({rot2})")

        # skip if too few tie points are found
        if tps.shape[0] < min_tps and tps.shape[0] < debug_min_tps:
            continue

        # limit number of tie points
        if tps.shape[0] > max_tps:

            # select the top points and conf
            top_indices = np.argsort(conf)[-max_tps:][::-1]
            tps = tps[top_indices]
            conf = conf[top_indices]

        # save the tie points and conf
        tp_dict[(img_1_id, img_2_id)] = tps
        conf_dict[(img_1_id, img_2_id)] = conf

    return tp_dict, conf_dict
