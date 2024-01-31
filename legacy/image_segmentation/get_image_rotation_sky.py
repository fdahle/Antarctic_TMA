import copy
import json
import numpy as np
import os

import base.print_v as p


def get_image_rotation_sky(segmented, image_id,
                           subset_height=None, sky_id=None,
                           catch=True, verbose=False, pbar=None):

    """
    get_image_rotation_sky(input_img, input_segmented, image_id, sky_id, return_bool, catch, verbose):
    This function checks the location of the class 'sky' in the segmented images.
    If the sky is on the top or the image is a vertical image, true is returned. Otherwise, false is
     returned. If no sky can be found in non-vertical images, an error is called.
    Args:
        segmented (np-array): The segmented version of the raw image.
        image_id (String): The image image_id of segmented.
        subset_height (int): How many pixels should be selected from the top and bottom
        sky_id (int): The number of the class 'sky' in the segmented image. It is usually '6',
            but can be changed.
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        corrected (Boolean, optional): True if the image is correct
    """

    p.print_v(f"Start: get_image_rotation_sky ({image_id})", verbose=verbose, pbar=pbar)

    # load the json to get default values
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    json_folder = project_folder + "/image_segmentation"
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if subset_height is None:
        subset_height = json_data["get_image_rotation_sky_subset_height"]

    if sky_id is None:
        sky_id = json_data["get_image_rotation_sky_sky_id"]

    segmented = copy.deepcopy(segmented)

    # no need to check the image if it is vertical
    if "V" in image_id:
        image_is_correct = True

    # we need to check the image
    else:

        # calculate percentages for top part of image
        uniques_top, counts_top = np.unique(segmented[0:subset_height, :],
                                            return_counts=True)
        percentages_top = dict(zip(uniques_top, counts_top * 100 /
                                   segmented[0:subset_height, :].size))

        # get the number of sky pixels in the subset (try, because it can be that sky is not in there)
        try:
            sky_top = percentages_top[sky_id]
            if sky_top < 1:
                sky_top = 0
        except (Exception,):
            sky_top = 0

        # calculate percentages for bottom part of image
        uniques_bottom, counts_bottom = np.unique(segmented[segmented.shape[0] - subset_height:, :],
                                                  return_counts=True)
        percentages_bottom = dict(zip(uniques_bottom, counts_bottom * 100 /
                                      segmented[segmented.shape[0] - subset_height:, :].size))

        # get the number of sky pixels in the subset (try, because it can be that sky is not in there)
        try:
            sky_bottom = percentages_bottom[sky_id]
            if sky_bottom < 0:
                sky_bottom = 0
        except (Exception,):
            sky_bottom = 0

        # image is right
        if sky_top > sky_bottom:
            image_is_correct = True
        elif sky_bottom > sky_top:
            image_is_correct = False
        else:
            if catch:
                p.print_v(f"Failed: get_image_rotation_sky ({image_id})", verbose=verbose, pbar=pbar)
                return None
            else:
                raise Exception("Something went wrong checking image orientation sky")

    p.print_v(f"Finished: get_image_rotation_sky ({image_id})", verbose=verbose, pbar=pbar)

    return image_is_correct
