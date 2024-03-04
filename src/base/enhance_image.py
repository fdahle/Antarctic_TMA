import copy
import numpy as np
import warnings

from scipy import ndimage
from skimage import exposure
from typing import Optional, Tuple


def enhance_image(image: np.ndarray,
                  mask: Optional[np.ndarray] = None,
                  scale: Tuple[float, float] = (0.0, 1.0),
                  mult: int = 255,
                  img_min: int = 0,
                  disksize: int = 3,
                  gamma: float = 1.25) -> np.ndarray:
    """
    Enhances the contrast of an image to improve the identification of more tie-points.
    This process involves applying a median filter to reduce noise, clipping the image intensities
    to specified quantiles, stretching the intensity range, and finally applying gamma correction.
    It is based on https://spymicmac.readthedocs.io/en/latest/tutorials/preprocessing/kh9_preprocessing.html
    Args:
        image (np.ndarray): The image to enhance.
        mask (Optional[np.ndarray]): An optional mask to exclude certain areas from consideration
            when calculating the image's intensity quantiles.
        scale (Tuple[float, float], optional): The minimum and maximum quantiles to which the
            image's intensities will be stretched. Defaults to (0.0, 1.0).
        mult (int, optional): The multiplier used to scale the stretched intensities to the desired
            range. Defaults to 255.
        img_min (int, optional): The minimum intensity value in the stretched image. Defaults to 0.
        disksize (int, optional): The size of the disk used for median filtering to reduce noise.
            Defaults to 3.
        gamma (float, optional): The gamma value for gamma correction, enhancing the image contrast.
            Defaults to 1.25.

    Returns:
        np.ndarray: The enhanced image.

    Notes:
        - This function is particularly useful for preprocessing images before feature detection,
          improving the visibility of relevant features.
        - The mask parameter can be used to focus enhancement on specific parts of the image,
          ignoring masked-out areas.
    """

    # Create a deep copy of the image to avoid modifying the original
    img = copy.deepcopy(image)

    # Apply a median filter to reduce noise
    filtered = ndimage.median_filter(img, size=disksize)

    # Calculate the max and min values based on the specified scale and mask
    if mask is None:
        max_val = np.nanquantile(filtered, max(scale))
        min_val = np.nanquantile(filtered, min(scale))
    else:
        # Apply the mask - this replaces non-masked values with NaN
        masked_filtered = np.where(mask, filtered, np.nan)
        max_val = np.nanquantile(masked_filtered, max(scale))
        min_val = np.nanquantile(masked_filtered, min(scale))

    # Clip the filtered image to the calculated max and min values
    filtered[filtered > max_val] = max_val
    filtered[filtered < min_val] = min_val

    # Stretch the image intensities to the full range and scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stretched = (mult * (filtered - min_val) / (max_val - min_val + img_min)).astype(np.uint8)

    # Apply gamma correction to enhance contrast
    adjusted = exposure.adjust_gamma(stretched, gamma=gamma)

    return adjusted
