"""Calculate a fiducial mark from a given image"""

# Library imports
import cv2
import numpy as np
import math
from typing import Optional

# Constants
MAX_GAP_LINE = 25
MIN_LENGTH_LINE = 25
RHO = 1
SUBSET_PERCENTAGE = 0.25
SUBSET_SIZE = (250, 250)
THETA = np.pi / 4  # the denominator tells in which steps are looked for lines
THRESHOLD = 15
TWEAK_VALS = [-20, 30, 10]


def calculate_fid_mark(image: np.ndarray, key: str,
                       subset_bounds: list[tuple[int, int, int, int]],
                       ) -> Optional[tuple[int, int]]:
    """
    Calculate a fiducial mark from a given image, based on the specified key. Here the fiducial marks
    in the corners (NE, NW, SE, SW) are calculated based on the lines we calculate for the subsets at
    the N, E, S, W positions.

    # Position of fid marks:
    # 3 7 2
    # 5   6
    # 1 8 4

    Args:
        image (np.ndarray): The input image from which the fiducial mark is to be calculated
        key (str): A two digit string ('ne', 'nw', 'se', 'sw') indicating the direction
        subset_bounds (List): A list with the two subsets that are required for the key
    Returns:
        Optional[tuple[int, int]]: A tuple containing the x and y coordinates of the calculated fiducial
            mark within the entire image. Returns None if no fiducial mark is detected.
    Raises:
        ValueError: If the provided `key` is not one of the expected values ('ne', 'nw', 'se', 'sw').
    """

    # check key
    if key not in ["ne", "nw", "se", "sw"]:
        raise ValueError("Key must be one of 'ne', 'nw', 'se', 'sw'.")

    # split the key in the two directions
    direction1 = list(key)[0]
    direction2 = list(key)[1]

    # check if we have the required subsets
    if subset_bounds[0] is None or subset_bounds[1] is None:
        return None

    # init required dicts
    mid_lines = {direction1: None, direction2: None}
    subsets = {direction1: None, direction2: None}
    subset_coordinates = {direction1: None, direction2: None}

    # we need to get vertical lines from the fid marks
    for i, sub_direction in enumerate([direction1, direction2]):

        # get coords from subset_bounds
        min_x, min_y, max_x, max_y = subset_bounds[i]

        # tweak the subsets so that lines are better recognizable
        if sub_direction in ["n", "s"]:
            min_x = min_x - 100
            max_x = max_x + 100
            min_y = min_y
            max_y = max_y
        elif sub_direction in ["e", "w"]:
            min_x = min_x
            max_x = max_x
            min_y = min_y - 100
            max_y = max_y + 100

        min_y = int(max(min_y, 0))
        max_y = int(min(max_y, image.shape[0]))
        min_x = int(max(min_x, 0))
        max_x = int(min(max_x, image.shape[1]))

        # extract the subset from the image
        subset = image[min_y:max_y, min_x:max_x]

        subset_coordinates[sub_direction] = [min_x, min_y, max_x, max_y]
        subsets[sub_direction] = subset

        # blur (to make the binarize easier) and binarize the subset (based on otsu, so the binarize is based on
        # the values in the subset
        subset_blurred = cv2.GaussianBlur(subset, (5, 5), 0)
        _, subset_binarized = cv2.threshold(subset_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
            continue

        # this list will save the lines with correct orientation
        correct_lines = []

        # iterate all the lines
        for line in all_lines:

            # get slope of line
            with np.errstate(divide='ignore'):
                line = line[0]
                slope = (line[3] - line[1]) / (line[2] - line[0])

            if sub_direction in ["n", "s"]:
                if slope == 0:
                    correct_lines.append(line)

            if sub_direction in ["e", "w"]:
                if math.isinf(slope):
                    correct_lines.append(line)

        # the line most closes to the middle is most probably the correct line;
        # and this time 2 lines, one in each part of the image to create one big line
        mid_line = None
        mid_line_1 = None  #
        mid_line_2 = None  #
        min_distance_1 = 2000000  # the distance to the middle (at the beginning set to a high value)
        min_distance_2 = 2000000  # the distance to the middle (at the beginning set to a high value)

        # get the middle of the image
        mid_of_image_x = int(subset.shape[1] / 2)
        mid_of_image_y = int(subset.shape[0] / 2)

        # get the closest line
        for line in correct_lines:

            # calculate distance
            distance = None  # so that ide is not complaining
            if sub_direction in ["n", "s"]:
                distance = np.abs(mid_y - line[1])
            elif sub_direction in ["e", "w"]:
                distance = np.abs(mid_x - line[0])

            # get the closest lines
            if sub_direction in ["n", "s"]:
                if line[0] < mid_of_image_x and distance < min_distance_1:  # noqa
                    min_distance_1 = distance
                    mid_line_1 = line
                if line[0] > mid_of_image_x and distance < min_distance_2:
                    min_distance_2 = distance
                    mid_line_2 = line
            elif sub_direction in ["e", "w"]:
                if line[1] < mid_of_image_y and distance < min_distance_1:  # noqa
                    min_distance_1 = distance
                    mid_line_1 = line
                if line[1] > mid_of_image_y and distance < min_distance_2:
                    min_distance_2 = distance
                    mid_line_2 = line

        # no line was found -> skip
        if mid_line_1 is not None and mid_line_2 is None:
            mid_line = mid_line_1
        elif mid_line_1 is None and mid_line_2 is not None:
            mid_line = mid_line_2
        elif mid_line_1 is None and mid_line_2 is None:
            continue
        else:
            # check if lines are almost identical
            diff = None
            if mid_line_1 is not None and mid_line_2 is not None:
                if sub_direction in ["n", "s"]:
                    diff = np.abs(mid_line_1[1] - mid_line_2[1])
                elif sub_direction in ["e", "w"]:
                    diff = np.abs(mid_line_1[0] - mid_line_2[0])

                if diff > 10:
                    continue

            # calculate a global line by using the innermost corners of each line
            # afterward, the line is [x1, y1, x2 ,y2]
            if sub_direction in ["n", "s"]:
                mid_line = [mid_line_1[0], mid_line_1[1], mid_line_2[2], mid_line_2[3]]  # noqa
            if sub_direction in ["e", "w"]:
                mid_line = [mid_line_1[2], mid_line_1[3], mid_line_2[0], mid_line_2[1]]  # noqa

        mid_lines[sub_direction] = mid_line

    # if one of the lines couldn't be found no corner can be calculated
    if mid_lines[direction1] is None or mid_lines[direction2] is None:
        return None

    # reproject lines from subject_coordinates to image_coordinates
    for sub_key in [direction1, direction2]:
        mid_line = mid_lines[sub_key]

        mid_line[0] = mid_line[0] + subset_coordinates[sub_key][0]
        mid_line[2] = mid_line[2] + subset_coordinates[sub_key][0]
        mid_line[1] = mid_line[1] + subset_coordinates[sub_key][1]
        mid_line[3] = mid_line[3] + subset_coordinates[sub_key][1]

    # calculate the corners
    x1 = mid_lines[direction1][0]
    x2 = mid_lines[direction1][2]
    y1 = mid_lines[direction1][1]
    y2 = mid_lines[direction1][3]
    x3 = mid_lines[direction2][0]
    x4 = mid_lines[direction2][2]
    y3 = mid_lines[direction2][1]
    y4 = mid_lines[direction2][3]

    # calculate the mark coordinates
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    px = int(px)
    py = int(py)

    fid_mark = (px, py)

    return fid_mark
