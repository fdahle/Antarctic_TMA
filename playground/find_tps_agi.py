import os
import numpy as np
from tqdm import tqdm

import src.base.find_overlapping_images as foi
import src.base.find_tie_points as ftp
import src.display.display_images as di
import src.load.load_image as li

debug_show_tps = False

def find_tps_agi(img_folder, mask_folder=None,
                            footprint_dict=None,
                            matching_method='all',
                            min_overlap=0.5,
                            min_conf=0.7,
                            min_tps=25, max_tps=4000):

    # Initialize lists and dictionaries
    image_ids = []       # List of image IDs
    combinations = []    # List of image combinations
    image_dict = {}      # Dictionary of loaded images
    mask_dict = {}       # Dictionary of loaded masks

    # Get the .tif images in the folder
    for filename in os.listdir(img_folder):
        if filename.lower().endswith('.tif'):
            image_ids.append(filename[:-4])  # Remove '.tif' extension

    # Create a list of all combinations of images based on the matching method
    if matching_method == 'all':
        # All possible combinations
        for i in range(len(image_ids)):
            for j in range(i+1, len(image_ids)):
                if image_ids[i] == image_ids[j]:
                    continue
                combinations.append((image_ids[i], image_ids[j]))
    elif matching_method == 'sequential':
        # Sequential combinations
        for i in range(len(image_ids)-1):
            combinations.append((image_ids[i], image_ids[i+1]))
    elif matching_method == 'overlap':
        # Overlapping images based on footprints
        footprints_lst = [footprint_dict[image_id] for image_id in image_ids]
        overlap_dict = foi.find_overlapping_images(image_ids, footprints_lst,
                                                   min_overlap=min_overlap)
        for img_id, overlap_lst in overlap_dict.items():
            for overlap_id in overlap_lst:
                if img_id == overlap_id:
                    continue
                combinations.append((img_id, overlap_id))
    else:
        raise NotImplementedError(f"Matching method {matching_method} not existing")

    # Initialize tie-point detector
    tpd = ftp.TiePointDetector('lightglue', verbose=True,
                               min_conf_value=min_conf)

    # List to hold the tie points in the desired format
    tie_points_data = []

    # Iterate over all combinations
    print("  Iterate combinations")
    for img_1_id, img_2_id in (pbar := tqdm(combinations)):
        # Load image 1
        if img_1_id not in image_dict:
            path_img1 = os.path.join(img_folder, img_1_id + ".tif")
            image_dict[img_1_id] = li.load_image(path_img1)
        image1 = image_dict[img_1_id]

        # Load image 2
        if img_2_id not in image_dict:
            path_img2 = os.path.join(img_folder, img_2_id + ".tif")
            image_dict[img_2_id] = li.load_image(path_img2)
        image2 = image_dict[img_2_id]

        # Load masks if available
        if mask_folder is not None:
            # Load mask 1
            if img_1_id not in mask_dict:
                path_mask1 = os.path.join(mask_folder, img_1_id + ".tif")
                path_mask1 = path_mask1.replace('.tif', '_mask.tif')
                mask_dict[img_1_id] = li.load_image(path_mask1)
            mask1 = mask_dict[img_1_id]

            # Load mask 2
            if img_2_id not in mask_dict:
                path_mask2 = os.path.join(mask_folder, img_2_id + ".tif")
                path_mask2 = path_mask2.replace('.tif', '_mask.tif')
                mask_dict[img_2_id] = li.load_image(path_mask2)
            mask2 = mask_dict[img_2_id]
        else:
            # No masks available
            mask1 = None
            mask2 = None

        # Get the tie points
        tps, conf = tpd.find_tie_points(image1, image2,
                                        mask1=mask1, mask2=mask2)

        pbar.set_postfix_str(f"{tps.shape[0]} tie points between "
                             f"{img_1_id} and {img_2_id}")

        # Skip if too few tie points are found
        if tps.shape[0] < min_tps:
            continue

        # Limit the number of tie points
        if tps.shape[0] > max_tps:
            # Select the top points based on confidence
            top_indices = np.argsort(conf)[-max_tps:][::-1]
            tps = tps[top_indices]
            conf = conf[top_indices]

        # Save the tie points in the desired format
        for i in range(tps.shape[0]):
            x1, y1, x2, y2 = tps[i]
            tie_point = {
                'projections': {
                    f'{img_1_id}.jpg': (x1, y1),
                    f'{img_2_id}.jpg': (x2, y2),
                }
            }
            tie_points_data.append(tie_point)

        if debug_show_tps:
            style_config = {
                'title': f"{tps.shape[0]} tie points between {img_1_id} and {img_2_id}",
            }
            di.display_images([image1, image2],
                              tie_points=tps, tie_points_conf=conf,
                              style_config=style_config)

    return tie_points_data
