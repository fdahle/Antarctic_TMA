"""rotate an image based on the position of a sidebar"""

# Library imports
import cv2
import numpy as np
import mahotas as mht

import src.display.display_images as di

DEBUG_SHOW_SIDEBARS = True

def correct_image_orientation(image: np.ndarray,
                              image_path: str) ->\
        bool:
    """
    This function determines whether an image needs to be rotated 180 degrees to correct
    its orientation. A correctly rotated image has the sidebar on the left.
    The decision is based on the comparison of homogeneity between the left and right sides of the
    image, as the sidebar has a lower homogeneity. If rotation is required,
    the image is rotated and saved back to the specified path.

    Args:
        image (np.ndarray): The image to check and possibly rotate.
        image_path (str): Path where the rotated image should be saved.

    Returns:
        bool: True if the image was rotated, False otherwise.
    """

    # first check if we need to rotate the image
    rotation_required = _check_sidebar(image)

    # image must be rotated
    if rotation_required:
        # rotate the image
        image = np.rot90(image, 2)

        # save the new rotated image
        cv2.imwrite(image_path, image)

    return rotation_required


def _check_sidebar(image: np.ndarray, subset_width: int = 300) -> bool:
    """
    This function extracts left and right subsets of the image, blurs and thresholds them to binary,
    and computes their texture homogeneity using the Haralick texture feature. The image is considered
    for rotation if the right sidebar is significantly more homogeneous than the left.

    Args:
        image (np.ndarray): The image to analyze.
        subset_width (int, optional): Width of the sidebars to analyze. Defaults to 300.

    Returns:
        bool: True if the right sidebar is more homogeneous and significantly different, indicating rotation is needed.
    """

    # by default no rotation
    must_rotate = False

    # get subset of east and west
    extracted_e = image[subset_width:image.shape[0] - subset_width, :subset_width]
    extracted_w = image[subset_width:image.shape[0] - subset_width, image.shape[1] - subset_width:image.shape[1]]

    # blur image
    blurred_e = cv2.GaussianBlur(extracted_e, (9, 9), 0)
    blurred_w = cv2.GaussianBlur(extracted_w, (9, 9), 0)

    # make binary
    _, blurred_e = cv2.threshold(blurred_e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, blurred_w = cv2.threshold(blurred_w, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if DEBUG_SHOW_SIDEBARS:
        di.display_images([blurred_e, blurred_w])

    # calculate homogeneity
    homog_e = mht.features.haralick(blurred_e).mean(0)[1]
    homog_w = mht.features.haralick(blurred_w).mean(0)[1]

    # calculate difference between homogeneity
    diff = abs(homog_w - homog_e)

    # check if we must rotate
    if homog_w > homog_e and diff > 0.1:
        must_rotate = True

    return must_rotate
