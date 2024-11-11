""" check if the sky is correctly positioned in an image """

import numpy as np


def check_sky(segmented, sky_id: int = 6,
              subset_height: int = 200,
              mask_padding: int = 700) -> \
        bool:
    """
    Check if the sky is correctly positioned in an image and returns true if the
    sky is correctly positioned at the top of the image. For both top and bottom a
    subset is checked for the number of sky pixels.
    Vertical images always are considered correct.
    Args:
        segmented (np.ndarray): The segmented image.
        sky_id (int): The id of the sky class. Defaults to 6.
        subset_height (int): The height of the subset to check. Defaults to 200.
        mask_padding (int): The padding of the mask & what will be removed from top
            and bottom before checking. Defaults to 700.
    Returns:
        bool: True if the sky is correctly positioned at the top of the image and
            otherwise false.
    """
    total_height = subset_height + mask_padding

    # calculate percentages for top part of image
    uniques_top, counts_top = np.unique(segmented[mask_padding:total_height, :],
                                        return_counts=True)
    percentages_top = dict(zip(uniques_top, counts_top * 100 /
                               segmented[mask_padding:total_height, :].size))

    # get the number of sky pixels in the subset (try, because it can be that sky is not in there)
    try:
        sky_top = percentages_top[sky_id]
        if sky_top < 1:
            sky_top = 0
    except (Exception,):
        sky_top = 0

    # calculate percentages for bottom part of image
    uniques_bottom, counts_bottom = np.unique(segmented[segmented.shape[0] - total_height:
                                                        segmented.shape[0] - mask_padding, :],
                                              return_counts=True)
    percentages_bottom = dict(zip(uniques_bottom, counts_bottom * 100 /
                                  segmented[segmented.shape[0] - total_height:
                                            segmented.shape[0] - mask_padding, :].size))

    # get the number of sky pixels in the subset (try, because it can be that sky is not in there)
    try:
        sky_bottom = percentages_bottom[sky_id]
        if sky_bottom < 0:
            sky_bottom = 0
    except (Exception,):
        sky_bottom = 0

    print("skyTOP, skyBOTTOM")
    print(sky_top, sky_bottom)

    # image is right
    if sky_top > sky_bottom:
        image_is_correct = True
    elif sky_bottom > sky_top:
        image_is_correct = False
    else:
        image_is_correct = True

    return image_is_correct
