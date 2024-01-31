import copy
import numpy as np

from skimage import exposure
from scipy import ndimage

# see https://spymicmac.readthedocs.io/en/latest/tutorials/preprocessing/kh9_preprocessing.html#contrast-enhancement


def enhance_image(image, mask=None, scale=(0.0, 1.0), mult=255, img_min=0,
                  disksize=3, qmin=0.02, qmax=0.98, gamma=1.25):

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

    stretched = (mult * (filtered - min_val) / (max_val - min_val + img_min)).astype(np.uint8)

    adjusted = exposure.adjust_gamma(stretched, gamma=gamma)

    return adjusted


if __name__ == "__main__":

    import base.load_image_from_file as liff
    import base.remove_borders as rb
    import base.resize_image as ri

    img_id = "CA181232V0065"

    img = liff.load_image_from_file(img_id)
    img = rb.remove_borders(img, img_id)
    img = ri.resize_image(img, (1000,1000))

    img = enhance_image(img, scale=(0.02, 0.98))

    import display.display_images as di
    di.display_images(img)