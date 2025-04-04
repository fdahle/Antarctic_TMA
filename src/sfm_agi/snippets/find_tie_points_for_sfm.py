"""find tie points in a set of images that can be used for sfm"""

# Python imports
import copy
import os

# Library imports
import psutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Local imports
import src.base.find_overlapping_images_geom as foi
import src.base.find_tie_points as ftp
import src.display.display_images as di
import src.load.load_image as li
import src.base.rotate_image as ri
import src.base.rotate_points as rp

# Constants
TPS_TXT_FLD = "/data/ATM/data_1/sfm/agi_data/tie_points"

# debug settings
debug_show_intermediate_steps = False
debug_show_final_tps = False


def find_tie_points_for_sfm(img_folder: str,
                            image_dims: dict[str, tuple[int, int]],
                            mask_folder: str | None =None,
                            footprint_dict: dict[str, np.ndarray] | None =None,
                            rotation_dict: dict[str, tuple[float, float]] | None = None,
                            matching_method: str = 'all',
                            tp_type: type = float,
                            min_overlap: float = 0.25,
                            step_range: int = 1,
                            max_step_range: int = 10,
                            min_conf: float = 0.7,
                            min_tps: int = 25,
                            max_tps: int = 4000,
                            use_cached_tps: bool = False,
                            save_tps: bool = False
                            ) -> (dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]):
    """
        Find tie points between images for Structure-from-Motion (SfM) processing.

        Args:
            img_folder (str): Path to the folder containing the images.
            image_dims (Dict[str, Tuple[int, int]]): Dictionary mapping image IDs to
                their dimensions.
            mask_folder (Optional[str]): Path to the folder containing masks for the images.
                Defaults to None.
            footprint_dict (Optional[Dict[str, np.ndarray]]): Dictionary mapping image IDs to footprint arrays.
                Defaults to None.
            rotation_dict (Optional[Dict[str, Tuple[float, float]]]): Dictionary mapping image IDs to rotation angles.
                Defaults to None.
            matching_method (str): Method to use for matching images ("all",
                "sequential", "overlap", "combined"). Defaults to "all".
            tp_type (type): Data type for tie points. Defaults to float.
            min_overlap (float): Minimum overlap ratio between images for matching.
                Defaults to 0.25.
            step_range (int): Range of sequential steps for image matching.
                Defaults to 1.
            max_step_range (int): Maximum range of steps for sequential matching.
                Defaults to 10.
            min_conf (float): Minimum confidence for tie points. Defaults to 0.7.
            min_tps (int): Minimum number of tie points required between image pairs.
                Defaults to 25.
            max_tps (int): Maximum number of tie points to retain per image pair.
                Defaults to 4000.
            use_cached_tps (bool): Whether to use cached tie points if available.
                Defaults to False.
            save_tps (bool): Whether to save tie points to cache files.
                Defaults to False.
    """

    # get all cached tps
    if use_cached_tps:
        cached_tps = [file.stem for file in Path(TPS_TXT_FLD).glob("*.txt")]
    else:
        cached_tps = []

    # retrieve .tif image files from the folder and filter based on dimensions
    image_ids = [filename[:-4] for filename in os.listdir(img_folder) if filename.lower().endswith('.tif')]
    image_ids = [img_id for img_id in image_ids if img_id in image_dims]

    # check if we have enough images
    if len(image_ids) < 2:
        raise ValueError(f"Not enough images ({len(image_ids)}) found for matching")

    # create list of combinations
    combinations = _create_combinations(image_ids, matching_method,
                                        footprint_dict, min_overlap,
                                        step_range, max_step_range)

    # init tie-point detector
    tpd = ftp.TiePointDetector('lightglue', verbose=True,
                               min_conf=min_conf, tp_type=tp_type,
                               display=debug_show_intermediate_steps)

    # dict for the tie points & confidence values
    tp_dict = {}
    conf_dict = {}

    # keep the latest image1 and mask1 in memory (need to keep track of the id)
    old_img_1_id = None
    orig_image1 = None
    orig_mask1 = None
    image1 = None
    mask1 = None

    # iterate over all combinations
    for idx, (img_1_id, img_2_id, _) in (pbar := tqdm(enumerate(combinations), total=len(combinations))):

        pbar.set_description("Find tie-points")
        process = psutil.Process()
        memory_usage = process.memory_info().rss  # in bytes

        # Load cached tie points if available
        if use_cached_tps:
            tps, conf, file_path = _load_cached_tps(TPS_TXT_FLD, img_1_id, img_2_id, cached_tps)
            if tps is not None:
                pbar.set_postfix_str(f"{tps.shape[0]} tps between {img_1_id} "
                                     f"and {img_2_id} (cached) - Memory: {memory_usage / 1024 ** 2:.2f} MB")
                if tps.shape[0] < min_tps or tps.shape[0]:
                    continue
                else:

                    # limit number of tie points
                    if tps.shape[0] > max_tps:
                        # select the top points and conf
                        top_indices = np.argsort(conf)[-max_tps:][::-1]
                        tps = tps[top_indices]
                        conf = conf[top_indices]

                    img_1_shape = image_dims[img_1_id]
                    img_2_shape = image_dims[img_2_id]

                    tp_error = False
                    if np.amax(tps[:, 0]) > img_1_shape[1] or np.amax(tps[:, 1]) > img_1_shape[0]:
                        print(f"Error cached tps for {img_1_id} (*) and {img_2_id}")
                        tp_error = True
                    if np.amax(tps[:, 2]) > img_2_shape[1] or np.amax(tps[:, 3]) > img_2_shape[0]:
                        print(f"Error cached tps for {img_1_id} and {img_2_id} (*)")
                        tp_error = True

                    if tp_error is False:
                        tp_dict[(img_1_id, img_2_id)] = tps
                        conf_dict[(img_1_id, img_2_id)] = conf
                        continue

        # load image 1
        if img_1_id != old_img_1_id:

            # free some memory
            del image1, mask1

            # load image 1
            path_img1 = os.path.join(img_folder, img_1_id + ".tif")
            image1 = li.load_image(path_img1)

            # load mask 1 if available
            if mask_folder is not None:
                path_mask1 = os.path.join(mask_folder, img_1_id + ".tif")
                path_mask1 = path_mask1.replace('.tif', '_mask.tif')
                mask1 = li.load_image(path_mask1)
            else:
                mask1 = None
                print(f"WARNING: No mask found for image 1 ({img_1_id})")

            # save a copy of the image
            orig_image1 = copy.deepcopy(image1)
            orig_mask1 = copy.deepcopy(mask1)

            # set the old image 1 id
            old_img_1_id = img_1_id

        # load image 2
        path_img2 = os.path.join(img_folder, img_2_id + ".tif")
        image2 = li.load_image(path_img2)

        # load mask 2
        if mask_folder is not None:
            path_mask2 = os.path.join(mask_folder, img_2_id + ".tif")
            path_mask2 = path_mask2.replace('.tif', '_mask.tif')
            mask2 = li.load_image(path_mask2)
        else:
            mask2 = None
            print(f"WARNING: No mask found for image 2 ({img_2_id})")

        # save a copy of the image
        orig_image2 = copy.deepcopy(image2)
        orig_mask2 = copy.deepcopy(mask2)

        # init rotation matrices
        rot1, rot_mat1 = 0, None
        rot2, rot_mat2 = 0, None

        # check if we have rotations
        if rotation_dict is None:
            pass

        # yeah, we have rotations
        else:
            # we only need rotations if both images are vertical
            if "V" in img_1_id and "V" in img_2_id:

                # get temporary the rotation angles
                tmp1 = rotation_dict[img_1_id][0]
                tmp2 = rotation_dict[img_2_id][0]

                # check difference in rotation
                rot_dif = abs(tmp1 - tmp2)

                # rotation is only required if both angles are very different
                if rot_dif > 15:

                    # save the rotation angles as well now
                    rot1 = tmp1
                    rot2 = tmp2

                    # rotate the images
                    image1, rot_mat1 = ri.rotate_image(orig_image1, rot1, return_rot_matrix=True)
                    image2, rot_mat2 = ri.rotate_image(orig_image2, rot2, return_rot_matrix=True)

                    # rotate the masks
                    if mask1 is not None:
                        mask1 = ri.rotate_image(orig_mask1, rot1)
                    if mask2 is not None:
                        mask2 = ri.rotate_image(orig_mask2, rot2)
                else:
                    rot1, rot2 = 0, 0
                    rotmat, rot_mat2 = None, None

        # get the tie points
        tps, conf = tpd.find_tie_points(image1, image2,
                                        mask1=mask1, mask2=mask2,
                                        image_id_1=img_1_id, image_id_2=img_2_id)

        # keep track if we rotated the images
        rotated_1 = False
        rotated_2 = False

        # keep the original rotation of image 1
        orig_rot1 = copy.deepcopy(rot1)
        orig_rotmat1 = copy.deepcopy(rot_mat1)

        # first try with 180 degrees rotation
        if tps.shape[0] < min_tps:

            # get the rotation of image 1
            rot1_180 = copy.deepcopy(rot1)

            # get image1 and mask1 again
            image1_180 = copy.deepcopy(orig_image1)
            if mask1 is not None:
                mask1_180 = copy.deepcopy(orig_mask1)

            # add 180 degrees to the rotation
            rot1_180 = (rot1_180 + 180) % 360

            # rotate image and mask
            image1_180, rot_mat1_180 = ri.rotate_image(image1_180, rot1_180,
                                                         return_rot_matrix=True)
            if mask1 is not None:
                mask1_180 = ri.rotate_image(mask1_180, rot1_180)

            # get the tie points
            tps_180, conf_180 = tpd.find_tie_points(image1_180, image2,
                                                    mask1=mask1_180, mask2=mask2)

            # check if we have more tie points
            if tps_180.shape[0] > tps.shape[0]:

                # keep track of the rotation
                rotated_1 = True

                # save the tps
                tps, conf = tps_180, conf_180
                rot1 = rot1_180
                rot_mat1 = rot_mat1_180

        # second try with 180 degrees rotation
        if tps.shape[0] < min_tps:

            # get the rotation of image 2
            rot2_180 = copy.deepcopy(rot2)

            # get image2 and mask2 again
            image2_180 = copy.deepcopy(orig_image2)

            if mask2 is not None:
                mask2_180 = copy.deepcopy(orig_mask2)

            # add 180 degrees to the rotation
            rot2_180 = (rot2_180 + 180) % 360

            # rotate image and mask
            image2_180, rot_mat2_180 = ri.rotate_image(image2_180, rot2_180,
                                                       return_rot_matrix=True)
            if mask2 is not None:
                mask2_180 = ri.rotate_image(mask2_180, rot2_180)

            # get the tie points
            tps_180, conf_180 = tpd.find_tie_points(image1, image2_180,
                                            mask1=mask1, mask2=mask2_180)

            # check if we have more tie points
            if tps_180.shape[0] > tps.shape[0]:

                # get back original rotation of image 1
                if rotated_1:
                    rot1 = orig_rot1
                    rot_mat1 = orig_rotmat1

                # keep track of the rotation
                rotated_2 = True

                # save the tps
                tps, conf = tps_180, conf_180
                rot2 = rot2_180
                rot_mat2 = rot_mat2_180

        # rotate points back
        if rot1 > 0 and tps.shape[0] > 0:
            tps[:, :2] = rp.rotate_points(tps[:, :2], rot_mat1, invert=True)
        if rot2 > 0 and tps.shape[0] > 0:
            tps[:, 2:] = rp.rotate_points(tps[:, 2:], rot_mat2, invert=True)

        # save the tie points as txt files for cache
        if save_tps:
            if img_1_id < img_2_id:

                np.savetxt(TPS_TXT_FLD + "/" + img_1_id + "_" + img_2_id + ".txt",
                           np.hstack((tps, conf[:, None])), fmt='%f')
            else:
                tps_switched = np.hstack((tps[:, 2:], tps[:, :2]))
                np.savetxt(TPS_TXT_FLD + "/" + img_2_id + "_" + img_1_id + ".txt",
                           np.hstack((tps_switched, conf[:, None])), fmt='%f')

        # skip if still too few tie points are found
        if tps.shape[0] < min_tps:
            continue

        if debug_show_final_tps:

            text1 = f"{img_1_id}"
            if rotated_1:
                text1 += " (R)"
            text2 = f"{img_2_id}"
            if rotated_2:
                text2 += " (R)"

            style_config = {
                'title': f"{tps.shape[0]} tie points between {text1} and {text2}",
            }

            if rotated_1 or rotated_2:

                di.display_images([image1, image2],
                                  tie_points=tps, tie_points_conf=conf,
                                  style_config=style_config)

        text1 = f"{img_1_id}"
        if rotated_1:
            text1 += " (R)"
        text2 = f"{img_2_id}"
        if rotated_2:
            text2 += " (R)"
        pbar.set_postfix_str(f"{tps.shape[0]} tps between {text1} "
                             f"and {text2} - Memory: {memory_usage / 1024 ** 2:.2f} MB")

        # limit number of tie points
        if tps.shape[0] > max_tps:
            # select the top points and conf
            top_indices = np.argsort(conf)[-max_tps:][::-1]
            tps = tps[top_indices]
            conf = conf[top_indices]

        if np.amax(tps[:,0]) > image_dims[img_1_id][1] or np.amax(tps[:,1]) > image_dims[img_1_id][0]:
            print(np.amax(tps[:,0]), np.amax(tps[:,1]))
            print(image_dims[img_1_id])
            raise ValueError(f"Error tps for {img_1_id} (*) and {img_2_id}")
        if np.amax(tps[:,0]) > image_dims[img_1_id][1] or np.amax(tps[:,1]) > image_dims[img_1_id][0]:
            print(np.amax(tps[:,2]), np.amax(tps[:,3]))
            print(image_dims[img_2_id])
            raise ValueError(f"Error tps for {img_1_id} and {img_2_id} (*)")

        # save the tie points and conf
        tp_dict[(img_1_id, img_2_id)] = tps
        conf_dict[(img_1_id, img_2_id)] = conf

        # free some memory
        del image2, mask2

        if idx == len(combinations) - 1:
            pbar.set_postfix_str("Finished!")

    return tp_dict, conf_dict

def _create_combinations(image_ids, matching_method,
                         footprint_dict, min_overlap,
                         step_range, max_step_range):
    # save the combinations of images in a list
    combinations = []

    # create a list with all combinations of images
    if matching_method == 'all':

        # All possible combinations
        combinations = [(image_ids[i], image_ids[j]) for i in range(len(image_ids)) for j in
                        range(i + 1, len(image_ids))]

    elif matching_method == 'sequential':

        # Create a dictionary to group images by (flight, direction)
        image_groups = {}
        for img_id in image_ids:
            flight = img_id[2:6]  # Extract flight number
            direction = img_id[6:9]  # Extract direction
            number = int(img_id[-4:])  # Convert last 4 digits to integer for easy comparisons
            key = (flight, direction)
            image_groups.setdefault(key, []).append((number, img_id))

        # For each (flight, direction) group, create sequential pairs based on step range
        for _, images in image_groups.items():
            # Sort images by their sequential number
            images.sort()
            numbers, ids = zip(*images)  # Separate numbers and IDs

            for i in range(len(numbers)):
                for step in range(1, step_range + 1):
                    # Check higher and lower steps within the group
                    if i + step < len(numbers):
                        combinations.append((ids[i], ids[i + step]))
                    if i - step >= 0:
                        combinations.append((ids[i], ids[i - step]))

    elif matching_method == 'overlap':

        # Find overlapping image pairs
        footprints_lst = [footprint_dict[image_id] for image_id in image_ids]
        overlap_dict = foi.find_overlapping_images_geom(image_ids, footprints_lst, min_overlap=min_overlap)

        for img_id, overlap_lst in overlap_dict.items():
            for overlap_id in overlap_lst:
                if img_id != overlap_id and (overlap_id, img_id) not in combinations:
                    combinations.append((img_id, overlap_id))

    # combine overlap and sequential and triplets
    elif matching_method == "combined":

        # Create a dictionary to group images by (flight, direction)
        image_groups = {}
        for img_id in image_ids:
            flight = img_id[2:6]  # Extract flight number
            direction = img_id[6:9]  # Extract direction
            number = int(img_id[-4:])  # Convert last 4 digits to integer for easy comparisons
            key = (flight, direction)
            image_groups.setdefault(key, []).append((number, img_id))

        # Combined matching: sequential + overlapping + triplets
        # 1. Sequential combinations with same flight and direction, within step range
        for _, images in image_groups.items():
            images.sort()
            numbers, ids = zip(*images)

            for i in range(len(numbers)):
                for step in range(1, step_range + 1):
                    if i + step < len(numbers):
                        combinations.append((ids[i], ids[i + step], 'sequential'))
                    if i - step >= 0:
                        combinations.append((ids[i], ids[i - step], 'sequential'))

        # 2. Add overlapping combinations
        footprints_lst = [footprint_dict[image_id] for image_id in image_ids]
        overlap_dict = foi.find_overlapping_images_geom(image_ids, footprints_lst, min_overlap=min_overlap)
        for img_id, overlap_lst in overlap_dict.items():
            for overlap_id in overlap_lst:
                if img_id != overlap_id and (overlap_id, img_id) not in combinations:
                    combinations.append((img_id, overlap_id, 'overlap'))

        # 3. Add triplets
        for image_id in image_ids:
            flight = image_id[2:6]
            direction = image_id[6:9]
            number = image_id[-4:]

            img_id_l = f"CA{flight}31L{number}"
            img_id_v = f"CA{flight}32V{number}"
            img_id_r = f"CA{flight}33R{number}"

            # Adding left, vertical, and right images based on direction
            if direction == "31L" and img_id_v in image_ids:
                overlap_dict.setdefault(image_id, []).append(img_id_v)
            elif direction == "32V":
                if img_id_l in image_ids:
                    overlap_dict.setdefault(image_id, []).append(img_id_l)
                if img_id_r in image_ids:
                    overlap_dict.setdefault(image_id, []).append(img_id_r)
            elif direction == "33R" and img_id_v in image_ids:
                overlap_dict.setdefault(image_id, []).append(img_id_v)

            # Add triplets to combinations
            triplet_combos = [(img_id_l, img_id_v), (img_id_v, img_id_r)]
            for pair in triplet_combos:
                if all(img in image_ids for img in pair) and (pair[1], pair[0]) not in combinations:
                    combinations.append((pair[0], pair[1], 'triplet'))

    else:
        raise NotImplementedError(f"Matching method {matching_method} not existing")

    # iterate over all combinations and apply the max_step_range
    new_combinations = []
    for img_1_id, img_2_id, overlap_type in combinations:
        flight1 = img_1_id[2:6]
        flight2 = img_2_id[2:6]
        direction1 = img_1_id[6:9]
        direction2 = img_2_id[6:9]
        number1 = int(img_1_id[-4:])
        number2 = int(img_2_id[-4:])

        # Check if the images are from the same flight path and within the max_step_range
        if (flight1 != flight2 or direction1 != direction2 or
                abs(number1 - number2) <= max_step_range):
            new_combinations.append((img_1_id, img_2_id, overlap_type))

    # remove potential duplicates (also reversed combinations)
    unique_combinations = set()
    for img1, img2, overlap_type in new_combinations:
        unique_combinations.add((tuple(sorted((img1, img2))), overlap_type))
    combinations = [(img1, img2, string_value) for (img1, img2), string_value in unique_combinations]

    # sort the combinations
    combinations = sorted(combinations)

    return combinations

def _load_cached_tps(tps_fld, img_1_id, img_2_id, cached_tps):

    # flag to check if we found a file
    tps_loaded = False

    # iterate all file combinations
    for file_name in [f"{img_1_id}_{img_2_id}", f"{img_2_id}_{img_1_id}"]:

        # iterate all files
        if file_name in cached_tps:

            # declare file path and check existence
            file_path = f"{tps_fld}/{file_name}.txt"
            if os.path.isfile(file_path) is False:
                continue

            # we found a file!
            tps_loaded = True

            # check the file size
            if os.stat(file_path).st_size > 0:
                data = np.atleast_2d(np.loadtxt(file_path))
                tps, conf = (data[:, 2:], data[:, :2]) \
                    if file_name == f"{img_2_id}_{img_1_id}" \
                    else (data[:, :4], data[:, 4])
            else:
                tps, conf = np.zeros((0, 4)), np.zeros(0)
            break

    # return the tps and conf if we found a file
    if tps_loaded:
        return tps, conf, file_path  # noqa
    else:
        return None, None, None