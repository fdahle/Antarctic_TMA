import os
import numpy as np
from tqdm import tqdm

import src.base.find_overlapping_images as foi
import src.base.find_tie_points as ftp
import src.display.display_images as di
import src.load.load_image as li

debug_show_tps = False


def find_tie_points_for_sfm(img_folder, mask_folder=None,
                            footprint_dict=None,
                            matching_method='all',
                            min_overlap=0.5,
                            min_conf=0.7,
                            min_tps=25, max_tps=4000):

    # init list and dicts for later usage
    image_ids = []  # list with image ids
    combinations = []  # list with image combinations
    image_dict = {}  # dict with the loaded images
    mask_dict = {}  # dict with the loaded masks

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
                               min_conf_value=min_conf)

    # dict for the tie points & confidence values
    tp_dict = {}
    conf_dict = {}

    # iterate over all combinations
    print("  Iterate combinations")
    for img_1_id, img_2_id in (pbar := tqdm(combinations)):

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

        # get the tie points
        tps, conf = tpd.find_tie_points(image1, image2,
                                        mask1=mask1, mask2=mask2)

        pbar.set_postfix_str(f"{tps.shape[0]} tie points between "
                             f"{img_1_id} and {img_2_id}")

        # skip if too few tie points are found
        if tps.shape[0] < min_tps:
            continue

        # limit number of tie points
        if tps.shape[0] > max_tps:

            # select the top points and conf
            top_indices = np.argsort(conf)[-max_tps:][::-1]
            tps = tps[top_indices]
            conf = conf[top_indices]

        # get ids from path
        id_img1 = img_1_id.split('.')[0]
        id_img2 = img_2_id.split('.')[0]

        # save the tie points and conf
        tp_dict[(id_img1, id_img2)] = tps
        conf_dict[(id_img1, id_img2)] = conf

        if debug_show_tps:
            style_config = {
                'title': f"{tps.shape[0]} tie points between {id_img1} and {id_img2}",
            }

            di.display_images([image1, image2],
                              tie_points=tps, tie_points_conf=conf,
                              style_config=style_config)


    return tp_dict, conf_dict
