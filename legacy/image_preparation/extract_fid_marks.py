import copy

import cv2 as cv2
import json
import numpy as np
import math
import os

import base.connect_to_db as ctd
import base.print_v as p

import display.display_images as di

debug_keys = ["n", "e", "s", "w"]  # for which keys do we want to see debug images
debug_show_subsets = False  #
debug_show_steps = False
debug_show_all_lines = False
debug_show_lines = False
debug_show_fid_subsets = False
debug_show_subset_marks = False  # show one mark in the subset
debug_show_corner_lines = False
debug_show_fid_marks = False  # show all marks in the big image


# Position of fid marks:
# 3 7 2
# 5   6
# 1 8 4


def extract_fid_marks(img, image_id, subset_data,
                      rho=None, threshold=None, min_length_line=None, max_gap_line=None, tweak_vals=None,
                      subset_height=None, subset_width=None, subset_percentage=None,
                      overwrite=True, catch=True, verbose=False, pbar=None):
    """
    extract_fid_marks(img, subset_data, catch):
    This function extracts eight different fiducial marks from images based on subsets. It is based on different
    methods from computer vision (e.g. line extraction). Note that the fid mark extraction is not always successful.
    Args:
        img (np-array): The image in which we want to extract fiducial marks.
        image_id (String): The image_id of the image for which we are extracting fid-marks
        subset_data(Dict): A dict that contains the exact position of the subsets for the image.
        rho (Integer): distance resolution in pixels of the Hough grid
        threshold (Integer): minimum number of votes (intersections in Hough grid cell)
        min_length_line (Integer): minimum number of pixels making up a line
        max_gap_line (Integer): maximum gap in pixels between connectable line segments
        tweak_vals (List): to tweak the values to even more the outer part
        subset_height (Integer): How many pixels does the subset have in height
        subset_width (Integer): How many pixels does the subset have in width
        subset_percentage (Float): how much of the subset are we using to get the mark
        overwrite (Boolean): If true, we don't extract fid-marks for already existing fid-marks
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        fid_marks (Dict): A dict with the x and y positions for each fid mark. The return values are for every mark
        a list with the x and y value which describes the pixel position of this mark based on the top left. If no
        mark could be found the value 'None' is returned for both values.
        Example: {'n': [2400, 3], 'ne': [5000, 4], 'e': None, 'se':[5000, 2500], ...}
    """

    p.print_v(f"Start: extract_fid_marks ({image_id})", verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the default value for rho
    if rho is None:
        rho = json_data["extract_fid_rho"]

    # get the default value for the threshold
    if threshold is None:
        threshold = json_data["extract_fid_threshold"]

    # get the default value for min_length_line
    if min_length_line is None:
        min_length_line = json_data["extract_fid_min_length_line"]

    # get the default value for max_gap_line
    if max_gap_line is None:
        max_gap_line = json_data["extract_fid_max_gap_line"]

    # get the default value for the tweak_vals
    if tweak_vals is None:
        tweak_vals = json_data["extract_fid_tweak_vals"]

    # get the default value for subset_percentage
    if subset_percentage is None:
        subset_percentage = json_data["extract_fid_subset_percentage"]

    if subset_height is None:
        subset_height = json_data["subset_height_px"]

    if subset_width is None:
        subset_width = json_data["subset_width_px"]

    try:
        # set the denominator tells in which steps are looked for lines (180 = 1 degree)
        theta = np.pi / 4

        # get already existing subset information from the db
        if overwrite is False:
            sql_string = f"SELECT * FROM images_fid_marks WHERE image_id='{image_id}'"
            table_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)
        else:
            table_data = None

        # check the image
        assert img is not None, "No image is loaded"

        # show the initial subsets on the image if wished
        if debug_show_subsets:

            subsets = []
            for direction in debug_keys:

                # check if the subset data is available for this key
                if subset_data[direction] is None:
                    continue

                # get coordinates of the subset
                min_x = int(subset_data[direction][0])
                max_x = int(min_x + subset_width)
                min_y = int(subset_data[direction][2])
                max_y = int(min_y + subset_height)

                # if min value is below 0 make to 0
                if min_x < 0:
                    min_x = 0
                if min_y < 0:
                    min_y = 0

                # if max value is bigger than image, make smaller
                if max_x > img.shape[1]:
                    max_x = img.shape[1]
                if max_y > img.shape[0]:
                    max_y = img.shape[0]

                subsets.append([min_x, min_y, max_x-min_x, max_y-min_y])

            di.display_images(img, bboxes=[subsets])

        conversion_dict = {
            "n": 7,
            "e": 6,
            "s": 8,
            "w": 5,
            "ne": 2,
            "se": 4,
            "nw": 3,
            "sw": 1
        }

        # store the fid-marks in this dict
        fid_marks = {}

        # first get the fid-marks on the sides of the images
        for direction in ["n", "e", "s", "w"]:

            # get the number of the key
            direction_nr = conversion_dict[direction]

            # init already the entry
            fid_marks[direction_nr] = None

            # if we already have the data (and it is not estimated) we don't need to extract data again
            if overwrite is False and \
                    table_data[f"fid_mark_{direction_nr}_x"].iloc[0] is not None and \
                    table_data[f"fid_mark_{direction_nr}_estimated"].iloc[0]:
                p.print_v(f"Fid-mark for {direction} for {image_id} already estimated", verbose,
                          "green", pbar=pbar)
                continue

            # init fid-mark for the beginning, so that if something goes wrong the backup fid mark is None
            fid_mark = None

            # check if the subset data is available for this key
            if subset_data[direction] is None:
                continue

            # get coordinates of the subset
            min_x = int(subset_data[direction][0])
            max_x = int(min_x + subset_width)
            min_y = int(subset_data[direction][2])
            max_y = int(min_y + subset_height)

            # if min value is below 0 make to 0
            if min_x < 0:
                min_x = 0
            if min_y < 0:
                min_y = 0

            # if max value is bigger than image, make smaller
            if max_x > img.shape[1]:
                max_x = img.shape[1]
            if max_y > img.shape[0]:
                max_y = img.shape[0]

            # extract the subset from the image
            subset = img[min_y:max_y, min_x:max_x]

            # blur (to make the binarize easier) and binarize the subset (based on otsu, so the binarize is based on
            # the values in the subset
            subset_blurred = cv2.GaussianBlur(subset, (5, 5), 0)
            _, subset_binarized = cv2.threshold(subset_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # apply canny edge detection
            subset_canny_edge = cv2.Canny(subset_binarized, 100, 100)

            if debug_show_steps and direction in debug_keys:
                di.display_images([subset, subset_blurred, subset_binarized, subset_canny_edge])

            """the following part is to extract lines from the images"""

            # get mid of image
            mid_x = int(subset_canny_edge.shape[1] / 2)
            mid_y = int(subset_canny_edge.shape[0] / 2)

            # get all lines
            all_lines = cv2.HoughLinesP(subset_canny_edge, rho, theta, threshold, np.array([]),
                                        min_length_line, max_gap_line)  # line is returned as [x0, y0, x1, y1]

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

                if direction in ["n", "s"]:
                    if math.isinf(slope):
                        correct_lines.append(line)

                if direction in ["e", "w"]:
                    if slope == 0:
                        correct_lines.append(line)

            if debug_show_all_lines and direction in debug_keys:
                di.display_images(subset, lines=correct_lines)

            # the line most closes to the middle is most probably the correct line
            mid_line = None  # the currently best line to the middle
            min_distance = 2000000  # the distance to the middle (at the beginning set to a high value)

            # get the closest line
            for line in correct_lines:

                # calculate distance
                distance = None  # so that ide is not complaining
                if direction in ["n", "s"]:
                    distance = np.abs(mid_x - line[0])
                elif direction in ["e", "w"]:
                    distance = np.abs(mid_y - line[1])

                # check distance
                if distance < min_distance:
                    mid_line = line
                    min_distance = distance

            # no line was found -> skip
            if mid_line is None:
                continue

            if debug_show_lines and direction in debug_keys:
                di.display_images(subset, lines=mid_line, title="middle line")

            """having the lines, it is possible to extract the fid-marks itself"""

            # get the avg line value
            avg_line_x = int((mid_line[0] + mid_line[2]) / 2)
            avg_line_y = int((mid_line[1] + mid_line[3]) / 2)

            # how many pixels next to the line are we searching for the fid-marks
            extra_search_width = 15

            # create an even smaller subset based on the line
            if direction in ["n", "s"]:
                min_x_sm = min_x + avg_line_x - extra_search_width
                max_x_sm = min_x + avg_line_x + extra_search_width

            elif direction in ["e", "w"]:
                min_y_sm = min_y + avg_line_y - extra_search_width
                max_y_sm = min_y + avg_line_y + extra_search_width

            for tweak_val in range(tweak_vals[0], tweak_vals[1], tweak_vals[2]):

                # only take the outer parts (there are the fid-marks)
                if direction == "n":
                    min_y_sm = min_y
                    max_y_sm = min_y + int(subset_height * subset_percentage)

                    min_y_sm = min_y_sm - tweak_val
                    max_y_sm = max_y_sm - tweak_val

                elif direction == "e":
                    min_x_sm = min_x + int(subset_width * (1 - subset_percentage))
                    max_x_sm = max_x

                    min_x_sm = min_x_sm + tweak_val
                    max_x_sm = max_x_sm + tweak_val

                elif direction == "s":
                    min_y_sm = min_y + int(subset_height * (1 - subset_percentage))
                    max_y_sm = max_y

                    min_y_sm = min_y_sm + tweak_val
                    max_y_sm = max_y_sm + tweak_val

                elif direction == "w":
                    min_x_sm = min_x
                    max_x_sm = min_x + int(subset_width * subset_percentage)

                    # the detection for west is not working completely right
                    # -> the fid-mark is usually more located to the west
                    min_x_sm = min_x_sm - 100
                    max_x_sm = max_x_sm - 100

                    min_x_sm = min_x_sm - tweak_val
                    max_x_sm = max_x_sm - tweak_val

                min_x_sm = max(0, min_x_sm)  # noqa
                min_y_sm = max(0, min_y_sm)  # noqa
                max_x_sm = min(img.shape[1], max_x_sm)  # noqa
                max_y_sm = min(img.shape[0], max_y_sm)  # noqa

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
                    fid_subset = img[min_y_sm:max_y_sm, min_x_sm:max_x_sm]  # noqa

                    if debug_show_fid_subsets and direction in debug_keys:
                        di.display_images([subset, fid_subset], list_of_types=["gray", "gray"])

                # get this smallest subset in which we think the fid-mark is
                fid_subset = img[min_y_sm:max_y_sm, min_x_sm:max_x_sm]  # noqa

                if debug_show_fid_subsets and direction in debug_keys:
                    di.display_images([subset, fid_subset], list_of_types=["gray", "gray"])

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
                    if direction in ["n", "s"]:
                        distance = np.abs(mid_of_image_x - mark[0])
                    elif direction in ["e", "w"]:
                        distance = np.abs(mid_of_image_y - mark[1])

                    # check distance
                    if distance < min_distance:
                        mid_mark = mark

                # no mark could fulfill the requirement
                if mid_mark is None:
                    continue

                if debug_show_subset_marks and direction in debug_keys:
                    di.display_images(fid_subset, points=mid_mark)

                mid_mark[0] = mid_mark[0] + min_x_sm
                mid_mark[1] = mid_mark[1] + min_y_sm

                # hurray we found a mark
                if mid_mark is not None:
                    fid_mark = mid_mark
                    break

            fid_marks[direction_nr] = fid_mark

        # Now get the marks at the corners
        for direction in ["ne", "se", "sw", "nw"]:

            # split the directions in the two components
            direction1 = list(direction)[0]
            direction2 = list(direction)[1]

            # get the number for the keys
            direction_nr = conversion_dict[direction]

            # init already the entry
            fid_marks[direction_nr] = None

            # if we already have the data (and it is not estimated) we don't need to extract data again
            if overwrite is False and \
                    table_data[f"fid_mark_{direction_nr}_x"].iloc[0] is not None and \
                    table_data[f"fid_mark_{direction_nr}_estimated"].iloc[0]:
                p.print_v(f"Fid-mark for {direction} for {image_id} already estimated", verbose,
                          "green", pbar=pbar)
                continue

            # check if we have the required subsets
            if subset_data[direction1] is None or subset_data[direction2] is None:
                continue

            mid_lines = {direction1: None, direction2: None}
            subsets = {direction1: None, direction2: None}
            subset_coordinates = {direction1: None, direction2: None}

            # we need to ger vertical lines from the fid marks
            for sub_direction in [direction1, direction2]:

                # get coordinates of the subset
                min_x = int(subset_data[sub_direction][0])
                max_x = int(min_x + subset_width)
                min_y = int(subset_data[sub_direction][2])
                max_y = int(min_y + subset_height)

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

                min_y = max(min_y, 0)
                max_y = min(max_y, img.shape[0])
                min_x = max(min_x, 0)
                max_x = min(max_x, img.shape[1])

                # extract the subset from the image
                subset = img[min_y:max_y, min_x:max_x]

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
                all_lines = cv2.HoughLinesP(subset_canny_edge, rho, theta, threshold, np.array([]),
                                            min_length_line, max_gap_line)  # line is returned as [x0, y0, x1, y1]

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
                    # afterwards line is [x1, y1, x2 ,y2]
                    if sub_direction in ["n", "s"]:
                        mid_line = [mid_line_1[0], mid_line_1[1], mid_line_2[2], mid_line_2[3]]  # noqa
                    if sub_direction in ["e", "w"]:
                        mid_line = [mid_line_1[2], mid_line_1[3], mid_line_2[0], mid_line_2[1]]  # noqa

                mid_lines[sub_direction] = mid_line

            if debug_show_corner_lines and direction in debug_keys:
                di.display_images([subsets[direction1], subsets[direction2]],
                                  lines=list(mid_lines.values()))

            # if one of the lines couldn't be found no corner can be calculated
            if mid_lines[direction1] is None or mid_lines[direction2] is None:
                continue

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
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

            px = int(px)
            py = int(py)

            fid_marks[direction_nr] = [px, py]

        if debug_show_fid_marks:
            di.display_images(img, points=[list(fid_marks.values())],
                              title=f"All fid marks for {image_id}")
    except (Exception,) as e:
        if catch:
            p.print_v(f"Failed: extract_fid_marks ({image_id})", verbose, pbar=pbar)
            return None
        else:
            raise e

    p.print_v(f"Finished: extract_fid_marks ({image_id})", verbose, pbar=pbar)

    return fid_marks
