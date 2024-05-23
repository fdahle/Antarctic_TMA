# Package imports
import copy
import cv2
import math
import numpy as np
from typing import Optional

# Display imports
import src.display.display_images as di

# Constants
MAX_GAP_LINE = 25
MIN_LENGTH_LINE = 25
RHO = 1
SUBSET_PERCENTAGE = 0.25
SUBSET_SIZE = (250, 250)
THETA = np.pi / 4  # the denominator tells in which steps are looked for lines
THRESHOLD = 15
TWEAK_VALS = [-20, 30, 10]


def extract_fid_mark(image: np.ndarray, key: str,
                     subset_bounds: tuple[int, int, int, int],
                     display: bool = False) -> Optional[tuple[int, int]]:
    """
    Extracts a fiducial mark from a given image, based on the specified key.
    This function applies a series of image processing steps (e.g., blurring, thresholding,
    canny edge detection) to extract lines and ultimately the fiducial marks at N, E, S, W positions
    from a specified subset of the input image.

    # Position of fid marks:
    # 3 7 2
    # 5   6
    # 1 8 4

    Args:
        image (np.ndarray): The input image from which the fiducial mark is to be extracted.
        key (str): A single character ('n', 'e', 's', 'w') indicating the direction (north, east,
            south, west) relative to which the fiducial mark is to be found.
        subset_bounds (Tuple[int, int, int, int]): A tuple containing the bounds
            (min_x, min_y, max_x, max_y) for the subset of the image to be processed.
        display (bool, optional): If True, the function will display the subset of the image used
            for fiducial mark extraction. Defaults to False.
    Returns:
        Optional[tuple[int, int]]: A tuple containing the x and y coordinates of the detected fiducial
            mark within the entire image. Returns None if no fiducial mark is detected.
    Raises:
        ValueError: If the provided `key` is not one of the expected values ('n', 'e', 's', 'w').
    """

    if key not in ["n", "e", "s", "w"]:
        raise ValueError("Key must be one of 'n', 'e', 's', 'w'.")

    # init empty fid mark
    fid_mark = None

    # get coords from subset_bounds
    min_x, min_y, max_x, max_y = subset_bounds

    # if min values are below 0 -> set to 0
    min_x = max(0, min_x)
    min_y = max(0, min_y)

    # if max values are above image shape -> set to image shape
    max_x = min(max_x, image.shape[1])
    max_y = min(max_y, image.shape[0])

    # extract the subset from the image
    subset = image[min_y:max_y, min_x:max_x]

    # blur (to make the binarize easier) and binarize the subset (based on otsu, so the binarize is based on
    # the values in the subset
    subset_blurred = cv2.GaussianBlur(subset, (5, 5), 0)
    _, subset_binarized = cv2.threshold(subset_blurred, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # apply canny edge detection
    subset_canny_edge = cv2.Canny(subset_binarized, 100, 100)

    """the following part is to extract lines from the images"""

    # get mid of image
    mid_x = int(subset_canny_edge.shape[1] / 2)
    mid_y = int(subset_canny_edge.shape[0] / 2)

    # get all lines
    all_lines = cv2.HoughLinesP(subset_canny_edge, RHO, THETA, THRESHOLD, np.array([]),
                                MIN_LENGTH_LINE, MAX_GAP_LINE)  # line is returned as [x0, y0, x1, y1]

    # no lines found :/
    if all_lines is None:
        return None

    # this list will save the lines with correct orientation
    correct_lines = []

    # iterate all the lines
    for line in all_lines:

        # get slope of line
        with np.errstate(divide='ignore'):
            line = line[0]
            slope = (line[3] - line[1]) / (line[2] - line[0])

        if key in ["n", "s"]:
            if math.isinf(slope):
                correct_lines.append(line)

        if key in ["e", "w"]:
            if slope == 0:
                correct_lines.append(line)

    # the line most closes to the middle is most probably the correct line
    mid_line = None  # the currently best line to the middle
    min_distance = 2000000  # the distance to the middle (at the beginning set to a high value)

    # get the closest line
    for line in correct_lines:

        # calculate distance
        distance = None  # so that ide is not complaining
        if key in ["n", "s"]:
            distance = np.abs(mid_x - line[0])
        elif key in ["e", "w"]:
            distance = np.abs(mid_y - line[1])

        # check distance
        if distance < min_distance:
            mid_line = line
            min_distance = distance

    # no line was found -> skip
    if mid_line is None:
        return None

    """having the lines, it is possible to extract the fid-marks itself"""

    # get the avg line value
    avg_line_x = int((mid_line[0] + mid_line[2]) / 2)
    avg_line_y = int((mid_line[1] + mid_line[3]) / 2)

    # how many pixels next to the line are we searching for the fid-marks
    extra_search_width = 15

    # init variables before loop
    min_x_sm, min_y_sm, max_x_sm, max_y_sm = 0, 0, 0, 0

    # create an even smaller subset based on the line
    if key in ["n", "s"]:
        min_x_sm = min_x + avg_line_x - extra_search_width
        max_x_sm = min_x + avg_line_x + extra_search_width
    elif key in ["e", "w"]:
        min_y_sm = min_y + avg_line_y - extra_search_width
        max_y_sm = min_y + avg_line_y + extra_search_width
    else:
        raise ValueError("Key must be one of 'n', 'e', 's', 'w'.")

    for tweak_val in range(TWEAK_VALS[0], TWEAK_VALS[1], TWEAK_VALS[2]):

        # only take the outer parts (there are the fid-marks)
        if key == "n":
            min_y_sm = min_y
            max_y_sm = min_y + int(SUBSET_SIZE[1] * SUBSET_PERCENTAGE)

            min_y_sm = min_y_sm - tweak_val
            max_y_sm = max_y_sm - tweak_val

        elif key == "e":
            min_x_sm = min_x + int(SUBSET_SIZE[0] * (1 - SUBSET_PERCENTAGE))
            max_x_sm = max_x

            min_x_sm = min_x_sm + tweak_val
            max_x_sm = max_x_sm + tweak_val

        elif key == "s":
            min_y_sm = min_y + int(SUBSET_SIZE[0] * (1 - SUBSET_PERCENTAGE))
            max_y_sm = max_y

            min_y_sm = min_y_sm + tweak_val
            max_y_sm = max_y_sm + tweak_val

        elif key == "w":
            min_x_sm = min_x
            max_x_sm = min_x + int(SUBSET_SIZE[1] * SUBSET_PERCENTAGE)

            # the detection for west is not working completely right
            # -> the fid-mark is usually more located to the west
            min_x_sm = min_x_sm - 100
            max_x_sm = max_x_sm - 100

            min_x_sm = min_x_sm - tweak_val
            max_x_sm = max_x_sm - tweak_val
        else:
            raise ValueError("Key must be one of 'n', 'e', 's', 'w'.")

        min_x_sm = max(0, min_x_sm)
        min_y_sm = max(0, min_y_sm)
        max_x_sm = min(image.shape[1], max_x_sm)
        max_y_sm = min(image.shape[0], max_y_sm)

        # take care for (random) negative values
        if min_x_sm > max_x_sm:
            temp = copy.deepcopy(min_x_sm)
            min_x_sm = max_x_sm
            max_x_sm = temp

        if min_y_sm > max_y_sm:
            temp = copy.deepcopy(min_y_sm)
            min_y_sm = max_y_sm
            max_y_sm = temp

        # get this smallest subset in which we think the fid-mark is
        fid_subset = image[min_y_sm:max_y_sm, min_x_sm:max_x_sm]

        # blur this subset
        fid_subset_blurred = cv2.GaussianBlur(fid_subset, (5, 5), 0)

        # get pixel value that is most common (=background color)
        ind = np.bincount(fid_subset_blurred.flatten()).argmax()
        thresh_val = ind + 10

        # threshold
        ret, fid_th = cv2.threshold(fid_subset_blurred, thresh_val, 255, cv2.THRESH_BINARY)

        # erode and dilate
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(fid_th, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        # find contours
        contours, hierarchy = cv2.findContours(dilated, 1, 2)

        all_marks = []
        for elem in contours:
            m = cv2.moments(elem)

            # get size
            size = m["m00"]

            # invalid contour
            if size == 0:
                continue

            # position
            c_x = int(m["m10"] / m["m00"])
            c_y = int(m["m01"] / m["m00"])

            fid_mark_min_size = 8
            fid_mark_max_size = 100

            # everything too big or too small is probably not a circle -> but save the rest
            if fid_mark_min_size < size < fid_mark_max_size:
                all_marks.append([c_x, c_y])

        # if no fid-marks are found: continue
        if len(all_marks) == 0:
            continue

        # if more than one mark is found take the one closest to the middle-line
        min_distance = 2000000
        mid_mark = None

        # get the middle of the image
        mid_of_image_x = int(dilated.shape[1] / 2)
        mid_of_image_y = int(dilated.shape[0] / 2)

        for mark in all_marks:

            # calculate distance
            distance = None
            if key in ["n", "s"]:
                distance = np.abs(mid_of_image_x - mark[0])
            elif key in ["e", "w"]:
                distance = np.abs(mid_of_image_y - mark[1])

            # check distance
            if distance < min_distance:
                mid_mark = mark

        # no mark could fulfill the requirement
        if mid_mark is None:
            continue

        # make the mark absolute
        mid_mark[0] = mid_mark[0] + min_x_sm
        mid_mark[1] = mid_mark[1] + min_y_sm

        # hurray we found a mark
        if mid_mark is not None:
            fid_mark = tuple(mid_mark)
            break

    if display and fid_mark is not None:
        fid_mark_subset_x = fid_mark[0] - min_x
        fid_mark_subset_y = fid_mark[1] - min_y
        di.display_images(subset, points=[[(fid_mark_subset_x, fid_mark_subset_y)]])

    return fid_mark
