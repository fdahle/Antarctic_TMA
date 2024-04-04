# Package imports
import cv2 as cv2
import numpy as np
import mahotas as mht


def correct_image_orientation(image, image_path):

    # first check if we need to rotate the image
    rotation_required = _check_sidebar(image)

    # image must be rotated
    if rotation_required:
        # rotate the image
        image = np.rot90(image, 2)

        # save the new rotated image
        cv2.imwrite(image_path, image)

    return rotation_required


def _check_sidebar(image, subset_width=300):
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

    # calculate homogeneity
    homog_e = mht.features.haralick(blurred_e).mean(0)[1]
    homog_w = mht.features.haralick(blurred_w).mean(0)[1]

    # calculate difference between homogeneity
    diff = abs(homog_w - homog_e)

    # check if we must rotate
    if homog_w > homog_e and diff > 0.1:
        must_rotate = True

    return must_rotate
