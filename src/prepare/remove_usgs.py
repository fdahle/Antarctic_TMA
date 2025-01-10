""" remove the usgs logo from an image """

# Library imports
import copy
import numpy as np
#from paddleocr import PaddleOCR
from typing import Optional

# Constants
LOGO_HEIGHT = 350  # in px
USE_GPU = False   # should the GPU used for OCR

# variables
check_logo = False  # should a check be applied before removing the logo


def remove_usgs(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Removes the USGS logo from an image if detected in the specified bottom part
    of the image.
    Args:
        image (np.ndarray): The image from which to remove the USGS logo.
    Returns:
        image (Optional[np.ndarray]): The image with the removed logo-part or None
            if no logo is removed
    """

    # create copy of the image to not change the original
    image = copy.deepcopy(image)

    # we want to check if we need to remove the logo
    if check_logo:

        # get the subpart of the image where we expect the logo
        sub_part = image[-LOGO_HEIGHT, :]

        # init ocr
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False,
                        version='PP-OCR', use_gpu=USE_GPU)

        # apply text extraction on that part
        ocr_results = ocr.ocr(sub_part)

        # Aggregate extracted text
        text = " ".join(elem[1][0] for block in ocr_results for elem in block if block)

        # Determine if the logo should be removed based on presence of "usgs"
        remove_flag = "usgs" not in text.lower()

    # we don't want to check
    else:

        # flag is therefore always true
        remove_flag = True

    # if flag is true we can remove the logo
    if remove_flag:

        # remove the USGS logo
        image = image[0:image.shape[0] - LOGO_HEIGHT, :]

    else:

        # set image to None to hint that we didn't remove the logo
        image = None

    return image
