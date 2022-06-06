import cv2
import math

import numpy as np

import display_single_image as dsi
import display_multiple_images as dmi

"""
extract_fid_marks(img, subset_data, catch):
This function extracts eight diffent fiducial points from images based on subsets. It is based on different methods
from computer vision (e.g line extraction). Note that the fid point extraction is not always successful.
INPUT:
    img (np-array): The image in which we want to extract fiducial points.
    subset_data(Dict): A dict that contains the exact position of the subsets for the image.
    catch (Boolean, True): If true and somethings is going wrong, the operation will continue and not crash.
        In this case None is returned
OUTPUT:
    fid_points (Dict): A dict with the x and y positions for each fid point. The return values are for every point
    a list with the x and y value which describes the pixel position of this point based on the top left. If no point could be
    found the value 'None' is returned for both values.
    Example: {'n': [2400, 3], 'ne': [5000, 4], 'e': None, 'se':[5000, 2500], ...}
"""



def extract_fid_marks(img, subset_data, catch=True):

    debug_show_steps = False
    debug_show_lines = False
    debug_show_fid_subsets = False
    debug_show_subset_points = False  # show one point in the subset
    debug_show_corner_lines = False
    debug_show_fid_points = False  # show all points in the big image

    # default params for line extraction
    rho = 1  # distance resolution in pixels of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 25  # minimum number of pixels making up a line
    max_line_gap = 25  # maximum gap in pixels between connectable line segments
    theta = np.pi / 4  # the denominator tells in which steps are looked for lines (180 = 1 degree)

    # catch empty subsets
    if subset_data.shape[0] == 0:
        if catch:
            return None
        else:
            raise ValueError("The subset data is empty")

    # check if subset_width and height are available in the subset data
    assert "subset_width" in subset_data.keys() and subset_data["subset_width"] is not None, \
        "subset_width must be available"
    assert "subset_height" in subset_data.keys() and subset_data["subset_height"] is not None, \
        "subset_height must be available"

    # store the fid points in this dict
    fid_points = {}

    # first get the fid points on the sides of the images
    for direction in ["n", "e", "s", "w"]:

        # init fid point for the beginning, so that if something goes wrong the backup fid point is None
        fid_point = None

        # check if the subset data is available for this direction
        if f"subset_{direction}_x" not in subset_data.keys() or f"subset_{direction}_y" not in subset_data.keys():
            continue
        if subset_data[f"subset_{direction}_x"] is None or subset_data[f"subset_{direction}_y"] is None:
            continue

        # get coordinates of the subset
        min_x = int(subset_data[f"subset_{direction}_x"])
        max_x = int(min_x + subset_data["subset_width"])
        min_y = int(subset_data[f"subset_{direction}_y"])
        max_y = int(min_y + subset_data["subset_height"])

        # if value is below 0 make to 0
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0

        # extract the subset from the image
        subset = img[min_y:max_y, min_x:max_x]

        # blur (to make the binarize easier) and binarize the subset (based on otsu, so the binarize is based on
        # the values in the subset
        subset_blurred = cv2.GaussianBlur(subset, (5, 5), 0)
        _, subset_binarized = cv2.threshold(subset_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # apply canny edge detection
        subset_canny_edge = cv2.Canny(subset_binarized, 100, 100)

        if debug_show_steps:
            dmi.display_multiple_images([subset, subset_blurred, subset_binarized, subset_canny_edge])

        """the following part is to extract lines from the images"""

        # get mid of image
        mid_x = int(subset_canny_edge.shape[1]/2)
        mid_y = int(subset_canny_edge.shape[0]/2)

        # get all lines
        all_lines = cv2.HoughLinesP(subset_canny_edge, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)  # line is returned as [x0, y0, x1, y1]

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

        # no line was found -> skip
        if mid_line is None:
            continue

        if debug_show_lines:
            dsi.display_single_image(subset, line=mid_line)

        """having the lines, it is possible to extract the fid points itself"""

        # get the avg line value
        avg_line_x = int((mid_line[0] + mid_line[2])/2)
        avg_line_y = int((mid_line[1] + mid_line[3])/2)

        # how many pixels next to the line are we searching for the fid-points
        extra_search_width = 15

        # create an even smaller subset based on the line
        if direction in ["n", "s"]:
            min_x_sm = min_x + avg_line_x - extra_search_width
            max_x_sm = min_x + avg_line_x + extra_search_width

        elif direction in ["e", "w"]:
            min_y_sm = min_y + avg_line_y - extra_search_width
            max_y_sm = min_y + avg_line_y + extra_search_width

        tweak_vals = [-10, 30, 10]  # to tweak the values to even more the outer part
        subset_percentage = .25  # how much of the subset are we using to get the point

        for tweak_val in range(tweak_vals[0], tweak_vals[1], tweak_vals[2]):

            # only take the outer parts (there are the fid points)
            if direction == "n":
                min_y_sm = min_y
                max_y_sm = min_y + int(subset_data["subset_height"]*subset_percentage)

                min_y_sm = min_y_sm - tweak_val
                if min_y_sm < 0:
                    min_y_sm = 0
                max_y_sm = max_y_sm - tweak_val

            elif direction == "e":
                min_x_sm = min_x + int(subset_data["subset_width"]*(1-subset_percentage))
                max_x_sm = max_x

                min_x_sm = min_x_sm + tweak_val
                if min_x_sm < 0:
                    min_x_sm = 0
                max_x_sm = max_x_sm + tweak_val

            elif direction == "s":
                min_y_sm = min_y + int(subset_data["subset_height"]*(1-subset_percentage))
                max_y_sm = max_y

                min_y_sm = min_y_sm + tweak_val
                max_y_sm = max_y_sm + tweak_val
                if max_y_sm >= img.shape[0]:
                    max_y_sm = img.shape[0]
                    min_y_sm = max_y_sm - 25

            elif direction == "w":
                min_x_sm = min_x
                max_x_sm = min_x + int(subset_data["subset_width"]*subset_percentage)

                # the detection for west is not working completely right
                # -> the fid point is usually more located to the west
                min_x_sm = min_x_sm - 100
                max_x_sm = max_x_sm - 100

                min_x_sm = min_x_sm - tweak_val
                max_x_sm = max_x_sm - tweak_val
                if max_x_sm >= img.shape[1]:
                    max_x_sm = img.shape[1]
                    min_x_sm = max_x_sm - 25

            if min_x_sm < 0:  # noqa
                min_x_sm = 0
            if min_y_sm < 0:  # noqa
                min_y_sm = 0

            # get this smallest subset in which we think the point is
            fid_subset = img[min_y_sm:max_y_sm, min_x_sm:max_x_sm]  # noqa

            if debug_show_fid_subsets:
                dmi.display_multiple_images([subset, fid_subset], list_of_types=["gray", "gray"])

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

            all_points = []
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

                fid_point_min_size = 8
                fid_point_max_size = 100

                # everything too big or too small is probably not a circle -> but save the rest
                if fid_point_min_size < size < fid_point_max_size:
                    all_points.append([c_x, c_y])

            # if no points are found continue
            if len(all_points) == 0:
                continue

            # if more than one point is found take the one closest to the middle-line
            min_distance = 2000000
            mid_point = None

            # get the middle of the image
            mid_of_image_x = int(dilated.shape[1] / 2)
            mid_of_image_y = int(dilated.shape[0] / 2)

            for point in all_points:

                # calculate distance
                distance = None
                if direction in ["n", "s"]:
                    distance = np.abs(mid_of_image_x - point[0])
                elif direction in ["e", "w"]:
                    distance = np.abs(mid_of_image_y - point[1])

                # check distance
                if distance < min_distance:
                    mid_point = point

            # no point could fulfill the requirement
            if mid_point is None:
                continue

            if debug_show_subset_points:
                dsi.display_single_image(fid_subset, point=mid_point)

            mid_point[0] = mid_point[0] + min_x_sm
            mid_point[1] = mid_point[1] + min_y_sm

            # hurray we found a point
            if mid_point is not None:
                fid_point = mid_point
                break

        fid_points[direction] = fid_point

    # Now get the points at the corners
    for direction in ["ne", "se", "sw", "nw"]:

        # split the directions in the two components
        direction1 = list(direction)[0]
        direction2 = list(direction)[1]

        # check if we have the required subsets
        if f"subset_{direction1}_x" not in subset_data.keys() or f"subset_{direction1}_y" not in subset_data.keys():
            continue
        if subset_data[f"subset_{direction1}_x"] is None or subset_data[f"subset_{direction1}_y"] is None:
            continue
        if f"subset_{direction2}_x" not in subset_data.keys() or f"subset_{direction2}_y" not in subset_data.keys():
            continue
        if subset_data[f"subset_{direction2}_x"] is None or subset_data[f"subset_{direction2}_y"] is None:
            continue

        mid_lines = {direction1: None, direction2: None}
        subsets = {direction1: None, direction2: None}
        subset_coordinates = {direction1: None, direction2: None}

        for sub_direction in [direction1, direction2]:

            # get coordinates of the subset
            min_x = int(subset_data[f"subset_{sub_direction}_x"])
            max_x = int(min_x + subset_data["subset_width"])
            min_y = int(subset_data[f"subset_{sub_direction}_y"])
            max_y = int(min_y + subset_data["subset_height"])

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

            if min_x < 0:
                min_x = 0
            if min_y < 0:
                min_y = 0

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
            mid_x = int(subset_canny_edge.shape[1]/2)
            mid_y = int(subset_canny_edge.shape[0]/2)

            # get all lines
            all_lines = cv2.HoughLinesP(subset_canny_edge, rho, theta, threshold, np.array([]),
                                        min_line_length, max_line_gap)  # line is returned as [x0, y0, x1, y1]

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

            # the line most closes to the middle is most probably the correct line; and this time 2 lines, on in each
            # part of the image to create one big line
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

        if debug_show_corner_lines:
            dmi.display_multiple_images([subsets[direction1], subsets[direction2]],
                                        lines=list(mid_lines.values()))

        # if one of the lines couldn't be found no corner can be calculated
        if mid_lines[direction1] is None or mid_lines[direction2] is None:
            continue

        # reproject lines from subject_coordinates to image_coordinates
        for sub_direction in [direction1, direction2]:

            mid_line = mid_lines[sub_direction]

            mid_line[0] = mid_line[0] + subset_coordinates[sub_direction][0]
            mid_line[2] = mid_line[2] + subset_coordinates[sub_direction][0]
            mid_line[1] = mid_line[1] + subset_coordinates[sub_direction][1]
            mid_line[3] = mid_line[3] + subset_coordinates[sub_direction][1]

        # calculate the corners
        x1 = mid_lines[direction1][0]
        x2 = mid_lines[direction1][2]
        y1 = mid_lines[direction1][1]
        y2 = mid_lines[direction1][3]
        x3 = mid_lines[direction2][0]
        x4 = mid_lines[direction2][2]
        y3 = mid_lines[direction2][1]
        y4 = mid_lines[direction2][3]

        # calculate the point coordinates
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        px = int(px)
        py = int(py)

        fid_points[direction] = [px, py]

    if debug_show_fid_points:
        dsi.display_single_image(img, point=list(fid_points.values()))

    return fid_points


if __name__ == "__main__":

    import get_ids_from_folder as giff
    import load_image_from_file as liff
    import connect_to_db as ctd

    path_folder_images = "/media/fdahle/beb5a64a-5335-424a-8f3c-779527060523/ATM/data/aerial/TMA/downloaded"

    ids = giff.get_ids_from_folder(path_folder_images, 10)

    for img_id in ids:
        image = liff.load_image_from_file(img_id)

        sql_string = "SELECT * FROM images_properties WHERE image_id='" + img_id + "'"
        data = ctd.get_data_from_db(sql_string)

        fid_marks = extract_fid_marks(image, data)

        print(fid_marks)

        break
