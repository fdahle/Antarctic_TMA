import copy
import cv2
import json
import numpy as np
import os

import base.print_v as p

# 1: ice, 2: snow, 3: rocks, 4: water, 5: clouds, 6:sky, 7: unknown


def mask_segmented(mask, segmented, values_to_mask=None, smooth_val=None,
                   catch=True, verbose=False, pbar=None):
    """
    mask_segmented(mask, segmented, values_to_mask, smooth_val, catch, verbose, pbar):
    This function takes an input mask and adds more masked values based on segmentation of the
    same image. 0 means masked, 1 means not masked. As the segmentation is very fine, but the
    mask should be more general, the mask is smoothed with the smoot-val
    Args:
        mask (np-array): The binary numpy-array containing the mask
        segmented (np-array): The segmented image
        values_to_mask (list): A list of the values of segmented, that should be masked
        smooth_val (int): How much are we smoothing the mask
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        mask (np-array): the new mask with the segmented information included
    """

    p.print_v("Start: mask_segmented", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if values_to_mask is None:
        values_to_mask = json_data["mask_segmented_mask_values"]

    if smooth_val is None:
        smooth_val = json_data["mask_segmented_smooth_val"]

    # create a deepcopy to not change the original mask
    temp_mask = copy.deepcopy(segmented)

    try:
        # mask the values
        for elem in values_to_mask:
            temp_mask[temp_mask == elem] = 0
        temp_mask[temp_mask > 0] = 1

        # smooth the segmented
        kernel = np.ones((smooth_val, smooth_val), np.uint8)
        temp_mask = cv2.dilate(temp_mask, kernel, iterations=1)  # noqa

        # add the segmented mask to the regular mask
        mask[temp_mask == 0] = 0

    except (Exception,) as e:
        if catch:
            p.print_v("Finished: mask_segmented", verbose=verbose, pbar=pbar)
            return None
        else:
            raise e

    p.print_v("Finished: mask_segmented", verbose=verbose, pbar=pbar)

    return mask
