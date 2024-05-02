# Package imports
import cv2
import dlib
import numpy as np
import math
from shapely.geometry import LineString

# Custom imports
import src.display.display_images as di
import src.estimate.altimeter_snippets as snippets

# Constants
CENTER_RADIUS_SIZE = 25
MIN_BINARY_THRESHOLD = 150
PATH_ALTIMETER_DETECTOR = "/data_1/ATM/data_1/machine_learning/dlib/altimeter/detector.svm"
PATH_TEMPLATES = "/data_1/ATM/data_1/machine_learning/dlib/altimeter/templates"

# Variables
matching_confidence_value = 20000000

# Debugging
debug_print = True
debug_show_altimeter = False
debug_show_binary = False
debug_show_circle = False
debug_show_long_lines = True
debug_show_selected_lines = True  # show center & tip lines
debug_show_pointer = True


def estimate_altimeter(image):

    if debug_print:
        print("Estimate altimeter for image")

    # detect altimeter subset
    altimeter = _detect_altimeter(image)

    # if no altimeter could be detected, return None
    if altimeter is None:
        return None

    if debug_show_altimeter:
        style_config = {'title': 'Altimeter subset'}
        di.display_images(altimeter, style_config=style_config)

    if debug_print:
        print("Altimeter detected successfully")

    # locate the circle in the altimeter
    circle = _locate_circle(altimeter)

    # if no circle could be located, return None
    if circle is None:
        return None

    if debug_show_circle:
        style_config = {'title': 'Located Circle', 'point_size': circle[2]}
        di.display_images(altimeter, points=[[(circle[0], circle[1])]], style_config=style_config)

    if debug_print:
        print(f"Circle located at {circle[0]} {circle[1]} with radius {circle[2]}")

    # find lines in the circle
    lines = _find_lines(altimeter)

    if lines is None:
        return None

    if debug_print:
        print(f"Found {len(lines)} lines in the altimeter subset")

    # select lines that are parallel or form a tip
    center_lines, tip_lines = _select_lines(circle, lines, altimeter)

    if debug_print:
        print(f"Found {len(center_lines)} center lines and {len(tip_lines)} tip lines")

    if debug_show_selected_lines:
        style_config = {'title': 'Center Lines (red) & Tip Lines (green)',
                        'line_color': ['red', 'green']}
        di.display_images(altimeter, lines=[[center_lines, tip_lines]],
                          style_config=style_config)

    # for altimeter detection we need both center and tip lines
    if center_lines is None or tip_lines is None:
        return None

    # get the parallel lines
    parallel_lines = _pair_lines(center_lines)

    # filter parallel lines
    correct_parallel_lines = _filter_parallel_lines(parallel_lines)

    # get height from line
    height = _lines2height(tip_lines, correct_parallel_lines, circle, altimeter)

    return height


def _detect_altimeter(image: np.ndarray):
    # load the object detector for altimeter detection
    detector = dlib.simple_object_detector(PATH_ALTIMETER_DETECTOR)

    # WARNING: Detector is trained on the complete image, so don't use a subset

    # detect where the altimeter is
    detections = detector(image)

    # check if there's only one detection (-> altimeter)
    if len(detections) != 1:
        return None

    # get the detection
    d = detections[0]

    # get the bounding box from the detection
    bounding_box = [d.left(), d.right(), d.top(), d.bottom()]

    # assure no values are outside the image
    if bounding_box[0] < 0:
        bounding_box[0] = 0
    if bounding_box[1] > image.shape[1]:
        bounding_box[1] = image.shape[1]
    if bounding_box[2] < 0:
        bounding_box[2] = 0
    if bounding_box[3] > image.shape[0]:
        bounding_box[3] = image.shape[0]

    # get the subset of the altimeter
    altimeter = image[bounding_box[2]:bounding_box[3], bounding_box[0]: bounding_box[1]]

    return altimeter


def _locate_circle(altimeter: np.ndarray):
    # load already the templates for 3, 5, 8
    template_3 = cv2.imread(PATH_TEMPLATES + '/3.jpg', 0)
    template_5 = cv2.imread(PATH_TEMPLATES + '/5.jpg', 0)
    template_8 = cv2.imread(PATH_TEMPLATES + '/8.jpg', 0)

    # get width and height of the 3 templates
    w3, h3 = template_3.shape[::-1]
    w5, h5 = template_5.shape[::-1]
    w8, h8 = template_8.shape[::-1]

    # set kernel for erode and dilate
    kernel = np.ones((3, 3), np.uint8)

    for min_th in range(MIN_BINARY_THRESHOLD, 250, 5):

        # get threshold for conversion to binary image
        ret, o1 = cv2.threshold(altimeter, min_th, 255, cv2.THRESH_BINARY)

        # erode and dilate the image
        eroded = cv2.erode(o1, kernel, iterations=2)
        dilated = cv2.dilate(eroded, kernel, iterations=2)

        # get binary
        altimeter_binary = 255 - dilated

        if debug_show_binary:
            style_config = {'title': 'Binary image'}
            di.display_images(altimeter_binary, style_config=style_config)

        # match image for 3
        res3 = cv2.matchTemplate(altimeter_binary, template_3, cv2.TM_CCOEFF)
        min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
        # matching for 5
        res5 = cv2.matchTemplate(altimeter_binary, template_5, cv2.TM_CCOEFF)
        min_val5, max_val5, min_loc5, max_loc5 = cv2.minMaxLoc(res5)
        # matching for 8
        res8 = cv2.matchTemplate(altimeter_binary, template_8, cv2.TM_CCOEFF)
        min_val8, max_val8, min_loc8, max_loc8 = cv2.minMaxLoc(res8)

        # locate the exact position of the circle by using the values of the template matching
        r = 168 + 25 + 74
        if max_val3 > matching_confidence_value and max_val3 > max_val5 and max_val3 > max_val8:
            top_left3 = max_loc3
            bottom_right3 = (top_left3[0] + w3, top_left3[1] + h3)
            mid_p = (int((top_left3[0] + bottom_right3[0]) / 2), int((top_left3[1] + bottom_right3[1]) / 2))
            circle = (
                int(mid_p[0] - np.cos(18 * np.pi / 180) * 205), int(mid_p[1] - np.sin(18 * np.pi / 180) * 205), r)
        elif max_val5 > matching_confidence_value and max_val5 > max_val3 and max_val5 > max_val8:
            top_left5 = max_loc5
            bottom_right5 = (top_left5[0] + w5, top_left5[1] + h5)
            circle = (int((top_left5[0] + bottom_right5[0]) / 2) - 7, top_left5[1] - 169, r)
        elif max_val8 > matching_confidence_value and max_val8 > max_val3 and max_val8 > max_val5:
            top_left8 = max_loc8
            bottom_right8 = (top_left8[0] + w8, top_left8[1] + h8)
            mid_p = (int((top_left8[0] + bottom_right8[0]) / 2), int((top_left8[1] + bottom_right8[1]) / 2))
            circle = (
                int(mid_p[0] + np.cos(18 * np.pi / 180) * 205), int(mid_p[1] + np.sin(18 * np.pi / 180) * 205), r)
        else:
            # no circle could be located
            return None

        return circle


def _find_lines(altimeter: np.ndarray):
    # get the edge of the binary image using Canny
    edges = cv2.Canny(image=altimeter, threshold1=100, threshold2=200, apertureSize=5)

    # HoughLinesP
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=28,  # Min number of votes for valid line
        minLineLength=16,  # Min allowed length of line
        maxLineGap=5  # Max allowed gap between line for joining them
    )

    # if we haven't found lines we need to return None
    if lines is None:
        return None

    # Iterate over lines for displaying
    lines_list = []
    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        lines_list.append([x1, y1, x2, y2])

    return lines_list


def _select_lines(circle, lines, altimeter):
    # get center of circle
    x_circle = circle[0]
    y_circle = circle[1]

    # we define a new radius that only covers the center
    r_circle = CENTER_RADIUS_SIZE

    # only keep lines that are going through the center of the circle
    selected_lines_center = []
    selected_lines_non_center = []
    for line in lines:

        # get coefficient of the line
        a = line[1] - line[3]
        b = line[2] - line[0]
        c = line[0] * line[3] - line[2] * line[1]

        # check if the line intersects with the circle
        d = np.abs(a * x_circle + b * y_circle + c) / np.sqrt(a ** 2 + b ** 2)
        if d <= r_circle:
            selected_lines_center.append(line)
        else:
            selected_lines_non_center.append(line)

    if debug_print:
        print(f"Lines separated into {len(selected_lines_center)} center "
              f"and {len(selected_lines_non_center)} non-center lines")

    # delete the lines that are too short (<25)
    selected_lines_center = snippets.delete_short_lines(selected_lines_center)
    selected_lines_non_center = snippets.delete_short_lines(selected_lines_non_center)

    if debug_print:
        print(f"{len(selected_lines_center)} center and "
              f"{len(selected_lines_non_center)} non-center lines "
              f"left after deletion of short lines")

    if debug_show_long_lines:
        style_config = {'title': 'Long lines for center (red) and non-center (green)',
                        'line_color': ['red', 'green']}
        print(selected_lines_center)
        di.display_images(altimeter,
                          lines=[[selected_lines_center, selected_lines_non_center]],
                          style_config=style_config)

    # merge lines together
    selected_lines_center = snippets.merge_lines(selected_lines_center)

    # look for lines that are making the tip of the short pointer
    selected_lines_tip = []
    selected_lines_tip_debug = []  # we need an extra list for showing the center lines
    for line_1 in selected_lines_non_center:

        # rearrange line so that the center point is the first point
        line_1 = snippets.rearrange_line(line_1, x_circle, y_circle)

        # don't use line again
        if line_1 in selected_lines_tip_debug:
            continue

        # we want to compare lines with other lines -> therefore second iteration
        for line_2 in selected_lines_non_center:

            # rearrange line so that the center point is the first point
            line_2 = snippets.rearrange_line(line_2, x_circle, y_circle)

            # we don't need to compare the same lines
            if line_1 == line_2:
                continue

            # don't use line again
            if line_1 in selected_lines_tip_debug or line_2 in selected_lines_tip_debug:
                continue

            # check if lines form a tip
            is_tip = snippets.do_lines_form_tip(line_1, line_2, x_circle, y_circle, circle[2])

            # save the lines if we have a tip
            if is_tip:
                selected_lines_tip.append([line_1, line_2])
                selected_lines_tip_debug.append(line_1)
                selected_lines_tip_debug.append(line_2)

    return selected_lines_center, selected_lines_tip


def _lines2height(lines_tip, lines_parallel, circle, altimeter: np.ndarray):
    if len(lines_tip) > 1:
        print("Too many lines tip found")
        return None

    if len(lines_parallel) > 2:
        print("Too many parallel lines found")
        return None

    final_lines = []

    # get middle line of short
    inter_x, inter_y = snippets.intersection_of_lines(lines_tip[0][0], lines_tip[0][1])
    middle_line_short = [circle[0], circle[1], inter_x, inter_y]
    final_lines.append(middle_line_short)

    # get middle line of long
    mid_x = int((lines_parallel[0][0] + lines_parallel[1][0]) / 2)
    mid_y = int((lines_parallel[0][1] + lines_parallel[1][1]) / 2)
    middle_line_long = [circle[0], circle[1], mid_x, mid_y]
    final_lines.append(middle_line_long)

    # get height value for short pointer
    middle_angle_short = snippets.get_angle(middle_line_short)
    reading_short = int(10000 / (2 * np.pi) * middle_angle_short / 1000) * 1000  # only need to know how many thousands

    # get height value for long pointer
    middle_angle_long = snippets.get_angle(middle_line_long)
    reading_long = int(1000 / (2 * np.pi) * middle_angle_long / 100) * 100

    height = 10000 + reading_short + reading_long

    if debug_show_pointer:
        di.display_images(altimeter, lines=final_lines)

    return height


def _pair_lines(lines_center):
    _parallel_lines = []

    for line1 in lines_center:

        for line2 in lines_center:

            # skip identical
            if line1 == line2:
                continue

            # skip already found lines
            skip_pair = False
            for pair in _parallel_lines:
                if line2 == pair[0] and line1 == pair[1]:
                    skip_pair = True
                    break
            if skip_pair:
                continue

            an = snippets.angle_between_lines(line1, line2)

            if an < 5:

                ls1 = LineString([(line1[0], line1[1]), (line1[2], line1[3])])
                ls2 = LineString([(line2[0], line2[1]), (line2[2], line2[3])])
                dist = ls1.distance(ls2)

                if dist < 25:
                    _parallel_lines.append([line1, line2])

    return _parallel_lines


def _filter_parallel_lines(parallel_lines):

    top_length = 0
    top_pair = None

    for para_line in parallel_lines:
        line1 = para_line[0]
        line2 = para_line[1]
        dist1 = math.sqrt((line1[2] - line1[0])**2 + (line1[3] - line1[1])**2)
        dist2 = math.sqrt((line2[2] - line2[0])**2 + (line2[3] - line2[1])**2)
        dist = np.mean([dist1, dist2])
        if dist > top_length:
            top_length = dist
            top_pair = para_line

    return top_pair