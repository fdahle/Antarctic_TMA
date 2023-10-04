import copy
import numpy as np
import warnings

from skimage import exposure
from scipy import ndimage


def enhance_image(image, mask=None, scale=(0.0, 1.0), mult=255, img_min=0,
                  disksize=3, gamma=1.25):
    """
    enhance_image(image, mask, scale, mult, img_min, disksize, q_min, q_max, gamma):
    This function enhances the contrast of an image in order that more tie-points can be found. It is based on
    # https://spymicmac.readthedocs.io/en/latest/tutorials/preprocessing/kh9_preprocessing.html#contrast-enhancement

    Args:
        image (np-array): The image we want to enhance
        mask (np-array, None): A mask that can be used to ignore areas when calculating max and min values of the image
            (useful if e.g. borders are in the image, as they would otherwise influence negatively the enhancement)
        scale (tuple, (0.0, 1.0): the minimum and maximum quantile to stretch to
        mult (int, 255): multiplier to scale the result to
        img_min (int, 0): What is the minimum value in the image
        disksize (int, 3): The size of the nan-median-filter
        gamma (float, 1.25): The value to use for the gamma adjustment

    Returns:
        adjusted (np-array): The enhanced image

    """
    img = copy.deepcopy(image)

    filtered = ndimage.median_filter(img, size=disksize)

    if mask is None:
        max_val = np.nanquantile(filtered, max(scale))
        min_val = np.nanquantile(filtered, min(scale))
    else:
        max_val = np.nanquantile(filtered[mask], max(scale))
        min_val = np.nanquantile(filtered[mask], min(scale))

    filtered[filtered > max_val] = max_val
    filtered[filtered < min_val] = min_val

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stretched = (mult * (filtered - min_val) / (max_val - min_val + img_min)).astype(np.uint8)

    adjusted = exposure.adjust_gamma(stretched, gamma=gamma)

    return adjusted


if __name__ == "__main__":
    import base.load_image_from_file as liff
    import base.remove_borders as rb
    import base.resize_image as ri

    img_id = "CA182632V0127"

    _img = liff.load_image_from_file(img_id)
    _img = rb.remove_borders(_img, img_id)
    _img = ri.resize_image(_img, (1000, 1000))

    _img = enhance_image(_img, scale=(0.02, 0.98))

    import display.display_images as di

    di.display_images(_img)
