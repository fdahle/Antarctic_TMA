"""extract an altimeter from an image"""

# Library imports
import sys
import copy
import cv2
import dlib
import numpy as np
import math
from shapely.geometry import LineString
from typing import Optional, Union

# Local imports
import src.display.display_images as di
import src.text.altimeter_snippets as snippets

# Constants
CENTER_RADIUS_SIZE = 25
MIN_BINARY_THRESHOLD = 150
PATH_ALTIMETER_DETECTOR = "/data/ATM/data_1/machine_learning/dlib/altimeter/detector.svm"
PATH_TEMPLATES = "/data/ATM/data_1/machine_learning/dlib/altimeter/templates"

# Variables
matching_confidence_value = 20000000

# Debugging
debug_print = True
debug_show_altimeter = False
debug_show_binary = False
debug_show_circle = False
debug_show_masked = False
debug_show_binarized = False
debug_show_all_lines = False
debug_show_long_lines = False
debug_show_center_lines = False
debug_show_special_lines = False  # show parallel & tip lines
debug_show_pointer = True


def extract_altimeter(image: np.ndarray,
                      return_position: bool = False) -> \
        Optional[Union[int, tuple[Optional[int], Optional[list[int]]]]]:
    """
    Extracts the altimeter reading from an image.

    Args:
        image (np.ndarray): The input image containing the altimeter.
        return_position (bool, optional): Whether to return the position of the detected altimeter. Defaults to False.

    Returns:
        Optional[Union[int, tuple[Optional[int], Optional[list[int]]]]]: The height reading from the altimeter,
        or None if no altimeter is detected. If return_position is True, returns a tuple containing the height and the
        bounding box of the detected altimeter.
    """

    if debug_print:
        print("Estimate altimeter for image")

    # detect altimeter subset
    altimeter, altimeter_box = _detect_altimeter(image)

    # if no altimeter could be detected, return None
    if altimeter is None:
        if debug_print:
            print("No altimeter detected")
        return None if not return_position else (None, None)

    if debug_show_altimeter:
        style_config = {'title': 'Altimeter subset'}
        di.display_images(altimeter, style_config=style_config)

    if debug_print:
        print("Altimeter detected successfully")

    # create the altimeter bounds
    bounding_box = [altimeter_box[0], altimeter_box[2],
                    altimeter_box[1] - altimeter_box[0],
                    altimeter_box[3] - altimeter_box[1]]

    # locate the circle in the altimeter
    circle, min_th = _locate_circle(altimeter)

    # if no circle could be located, return None
    if circle is None:
        if debug_print:
            print("No circle located in the altimeter subset")
        return None if not return_position else (None, bounding_box)

    if debug_show_circle:
        style_config = {'title': 'Located Circle', 'point_size': circle[2]}
        di.display_images(altimeter, points=[[(circle[0], circle[1])]], style_config=style_config)

    if debug_print:
        print(f"Circle located at {circle[0]} {circle[1]} with radius {circle[2]}")

    # set everything outside the circle to nan
    altimeter_masked = _mask_circle(altimeter, circle, 0)

    if debug_show_masked:
        di.display_images(altimeter_masked)

    # binarize the image with the circle
    altimeter_binarized = _binarize_circle(altimeter_masked, min_th + 20, 255)

    if debug_show_binarized:
        di.display_images([altimeter, altimeter_binarized])

    # find lines in the circle
    lines = _find_lines(altimeter_binarized)

    if lines is None:
        if debug_print:
            print("No lines found in the altimeter subset")
        return None if not return_position else (None, bounding_box)

    if debug_print:
        print(f"Found {len(lines)} lines in the altimeter subset")

    if debug_show_all_lines:
        style_config = {'title': 'All lines in altimeter subset'}
        di.display_images(altimeter, lines=[lines], style_config=style_config)

    # remove short lines
    lines = snippets.delete_short_lines(lines, 20)

    if lines is None:
        if debug_print:
            print("No lines left after deletion of short lines")
        return None if not return_position else (None, bounding_box)

    if debug_print:
        print(f"{len(lines)} lines left after deletion of short lines")

    if debug_show_long_lines:
        style_config = {'title': 'Long lines in altimeter subset'}
        di.display_images(altimeter, lines=[lines], style_config=style_config)

    # get center and non-enter lines
    center_lines, non_center_lines = _get_center_lines(circle, lines)

    if debug_print:
        print(f"Lines separated into {len(center_lines)} center "
              f"and {len(non_center_lines)} non-center lines")

    if debug_show_center_lines:
        style_config = {'title': 'Center (left) and non-center (right) lines',
                        'line_color': ['red', 'green']}
        di.display_images([altimeter, altimeter],
                          lines=[center_lines, non_center_lines],
                          style_config=style_config)

    if debug_print:
        print(f"Found {len(center_lines)} center lines and {len(non_center_lines)} non center lines")

    # select lines that are parallel or form a tip
    parallel_lines, tip_lines = _select_lines(altimeter, center_lines, non_center_lines, circle)

    if debug_print:
        print(f"Found {len(parallel_lines)} parallel lines and {len(tip_lines)} tip lines")

    if debug_show_special_lines:
        # flatten lines
        display_parallel_lines = [inner for sublist in parallel_lines for inner in sublist]
        display_tip_lines = [inner for sublist in tip_lines for inner in sublist]

        style_config = {'title': 'Parallel Lines (left) & Tip Lines (right)',
                        'line_color': ['red', 'green']}
        di.display_images([altimeter, altimeter],
                          lines=[display_parallel_lines, display_tip_lines],
                          style_config=style_config)

    # for altimeter detection we need both center and tip lines
    if (tip_lines is None or len(tip_lines)== 0):
        if debug_print:
            print("Parallel or tip lines is None")
        return None if not return_position else (None, bounding_box)

    # filter parallel lines
    if len(parallel_lines) > 0:
        parallel_lines = _filter_parallel_lines(parallel_lines)

    # get height from line
    height = _lines2height(tip_lines, parallel_lines, circle, altimeter)

    return height if not return_position else (height, bounding_box)


def _detect_altimeter(image: np.ndarray) -> tuple[Optional[np.ndarray], Optional[list[int]]]:
    """
    Detects the altimeter within the image using dlib.

    Args:
        image (np.ndarray): The input image.

    Returns:
        tuple[Optional[np.ndarray], Optional[list[int]]]: The detected altimeter image and its bounding box,
        or (None, None) if no altimeter is detected.
    """
    # load the object detector for altimeter detection
    detector = dlib.simple_object_detector(PATH_ALTIMETER_DETECTOR)

    # WARNING: Detector is trained on the complete image, so don't use a subset

    # detect where the altimeter is
    detections = detector(image)

    # check if there's only one detection (-> altimeter)
    if len(detections) != 1:
        return None, None

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

    return altimeter, bounding_box


def _locate_circle(altimeter: np.ndarray) -> tuple[Optional[tuple[int, int, int]], Optional[int]]:
    """
    Locates the circle in the altimeter image by finding the position of the numbers 3, 5, 8

    Args:
        altimeter (np.ndarray): The altimeter image.

    Returns:
        tuple[Optional[tuple[int, int, int]], Optional[int]]: The circle's center coordinates and radius,
        or (None, None) if no circle is located.
    """
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
            return None, None

        return circle, min_th

    return None, None


def _mask_circle(img: np.ndarray, circle: tuple[int, int, int], mask_val='auto') -> np.ndarray:
    """
    Masks everything outside of the detected circle with NaN.

    Args:
        img (np.ndarray): The input image.
        circle (tuple[int, int, int]): The detected circle as (x, y, radius).

    Returns:
        np.ndarray: The masked image with values outside the circle set to NaN.
    """
    # Create a copy of the image to avoid modifying the original
    masked_img = img.copy().astype(float)

    # Get the circle parameters
    x, y, radius = circle

    # Create a grid of coordinates
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]

    # Apply the circle equation (x-x0)^2 + (y-y0)^2 <= radius^2
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2

    # auto calc mask value
    if mask_val == 'auto':
        # get median of content inside the mask
        mask_val = np.nanmedian(img[mask])

    # Set values outside the circle to NaN
    masked_img[~mask] = mask_val


    return masked_img


def _find_lines(altimeter: np.ndarray) -> Optional[list[tuple[int, int, int, int]]]:
    """
    Finds lines in the altimeter image.

    Args:
        altimeter (np.ndarray): The altimeter image.

    Returns:
        Optional[list[tuple[int, int, int, int]]]: A list of detected lines, where each line
            is represented by its endpoints [x1, y1, x2, y2], or None if no lines are found.
    """
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


def _get_center_lines(circle: tuple[int, int, int],
                      lines: list[tuple[int, int, int, int]]) -> \
        tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """
    Separates the detected lines into center and non-center lines.

    Args:
        circle (tuple[int, int, int]): The circle's center coordinates and radius.
        lines (list[tuple[int, int, int, int]]): A list of detected lines.

    Returns:
        tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]: Two lists
            containing the center lines and non-center lines, respectively.
    """
    # get center of circle
    x_circle = circle[0]
    y_circle = circle[1]

    # we define a new radius that only covers the center
    r_circle = CENTER_RADIUS_SIZE

    # only keep lines that are going through the center of the circle
    lines_center = []
    lines_non_center = []
    for line in lines:

        # get coefficient of the line
        a = line[1] - line[3]
        b = line[2] - line[0]
        c = line[0] * line[3] - line[2] * line[1]

        # check if the line intersects with the circle
        d = np.abs(a * x_circle + b * y_circle + c) / np.sqrt(a ** 2 + b ** 2)
        if d <= r_circle:
            lines_center.append(line)
        else:
            lines_non_center.append(line)

    # merge lines together
    lines_center = snippets.merge_lines(lines_center)
    #lines_non_center = snippets.merge_lines(lines_non_center)

    return lines_center, lines_non_center


def _select_lines(altimeter,
        selected_lines_center: list[tuple[int, int, int, int]],
                  selected_lines_non_center: list[tuple[int, int, int, int]],
                  circle: tuple[int, int, int]) -> tuple[list[tuple[int, int, int, int]],
                                                         list[tuple[int, int, int, int]]]:
    """
    Selects lines that are parallel or form a tip from the detected lines.

    Args:
        selected_lines_center (list[tuple[int, int, int, int]]): A list of center lines.
        selected_lines_non_center (list[tuple[int, int, int, int]]): A list of non-center lines.
        circle (tuple[int, int, int]): The circle's center coordinates and radius.

    Returns:
        tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]: Two lists
            containing the parallel lines and tip lines, respectively.
    """
    # get center of circle
    x_circle = circle[0]
    y_circle = circle[1]

    # get parallel lines
    parallel_lines = _pair_lines(altimeter, selected_lines_center)

    # rearrange all lines once relative to the circle center
    rearranged_lines = [snippets.rearrange_line(line, x_circle, y_circle) for
                        line in selected_lines_non_center]

    # look for lines that are making the tip of the short pointer
    tip_lines = []
    for i, line_1 in enumerate(rearranged_lines):
        # Don't use line again if already used in a tip
        if any(line_1 in pair for pair in tip_lines):
            continue

        for line_2 in rearranged_lines[i + 1:]:  # Start from i+1 to avoid duplicate pairs
            # We don't need to compare the same lines
            if line_1 == line_2:
                continue

            # Check if these two lines form a tip
            if snippets.do_lines_form_tip(line_1, line_2, x_circle, y_circle, circle[2]):
                tip_lines.append([line_1, line_2])

    return parallel_lines, tip_lines


def _lines2height(lines_tip: list[tuple[int, int, int, int]],
                  lines_parallel: list[tuple[int, int, int, int]],
                  circle: tuple[int, int, int], altimeter: np.ndarray) -> Optional[int]:
    """
    Calculates the height from the detected lines and the circle.

    Args:
        lines_tip (list[tuple[int, int, int, int]]): A list of tip lines.
        lines_parallel (list[tuple[int, int, int, int]]): A list of parallel lines.
        circle (tuple[int, int, int]): The circle's center coordinates and radius.
        altimeter (np.ndarray): The altimeter image.

    Returns:
        Optional[int]: The calculated height, or None if the height could not be determined.
    """
    if len(lines_tip) > 1:
        print("Too many lines tip found")
        return None

    if len(lines_parallel) > 2:
        print("Too many parallel lines found")
        return None

    # get middle line of short
    inter_x, inter_y = snippets.intersection_of_lines(lines_tip[0][0], lines_tip[0][1])
    middle_line_short = [circle[0], circle[1], int(inter_x), int(inter_y)]

    print(lines_parallel)

    # get middle line of long
    if len(lines_parallel) == 2:
        mid_x = int((lines_parallel[0][0] + lines_parallel[1][0]) / 2)
        mid_y = int((lines_parallel[0][1] + lines_parallel[1][1]) / 2)
        middle_line_long = [circle[0], circle[1], mid_x, mid_y]

    # get height value for short pointer
    middle_angle_short = snippets.get_angle(middle_line_short)
    reading_short = int(10000 / (2 * np.pi) * middle_angle_short / 1000) * 1000  # only need to know how many thousands

    # get height value for long pointer
    if len(lines_parallel) == 2:
        middle_angle_long = snippets.get_angle(middle_line_long)
        reading_long = int(1000 / (2 * np.pi) * middle_angle_long / 100) * 100
    else:
        reading_long = 0
        middle_line_long = None

    height = 10000 + reading_short + reading_long

    # it is likely that a height below 8000 is above 10000
    if height < 8000:
        height += 10000

    if debug_show_pointer:
        print(middle_line_long)
        style_config = {"title": 'Short Pointer (left) & Long Pointer (right)'}
        di.display_images([altimeter],
                          lines=[[middle_line_short, middle_line_long]],
                          style_config=style_config)

    return height


def _pair_lines(altimeter,
                lines: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """
    Identifies pairs of parallel lines from the detected lines.

    Args:
        lines (list[list[int]]): A list of detected lines.

    Returns:
        list[list[int]]: A list of parallel lines.
    """
    parallel_lines = []

    for i, line1 in enumerate(lines):
        for line2 in lines[i + 1:]:  # Start from the next index to avoid duplicate pairs and self-comparison

            # Calculate the angle between the two lines
            angle = snippets.angle_between_lines(line1, line2)

            # Check if lines are sufficiently parallel
            if angle < 5:
                # Create LineString objects for more complex geometric operations
                ls1 = LineString([(line1[0], line1[1]), (line1[2], line1[3])])
                ls2 = LineString([(line2[0], line2[1]), (line2[2], line2[3])])

                # Calculate the distance to check how close the lines are
                distance = ls1.distance(ls2)

                # If lines are close enough, consider them as parallel
                if distance < 50:
                    parallel_lines.append([line1, line2])

            #style_config={'title': f'angle: {angle}, distance: {distance}'}
            #di.display_images([altimeter], lines=[[line1, line2]], style_config=style_config)

    return parallel_lines


def _filter_parallel_lines(parallel_lines: list[list[list[int]]]) -> list[list[int]]:
    """
    Filters the parallel lines to find the longest pair.

    Args:
        parallel_lines (list[list[list[int]]]): A list of parallel lines.

    Returns:
        list[list[int]]: The longest pair of parallel lines.
    """
    top_length = 0
    top_pair = []

    for para_line in parallel_lines:
        line1 = para_line[0]
        line2 = para_line[1]
        dist1 = math.sqrt((line1[2] - line1[0]) ** 2 + (line1[3] - line1[1]) ** 2)
        dist2 = math.sqrt((line2[2] - line2[0]) ** 2 + (line2[3] - line2[1]) ** 2)
        dist = np.mean([dist1, dist2])
        if dist > top_length:
            top_length = dist
            top_pair = para_line

    return top_pair


def _binarize_circle(img: np.ndarray, min_th: int = 0, max_th: int = 255) -> np.ndarray:
    """
    Binarize the image within the detected circle.

    Args:
        img (np.ndarray): The input image.
        min_th (int, optional): Minimum threshold for binarization. Defaults to 0.
        max_th (int, optional): Maximum threshold for binarization. Defaults to 255.

    Returns:
        np.ndarray: The binarized image.
    """
    img = copy.deepcopy(img)

    # 3) Convert circle_region to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        # e.g. it might be float32 or something else; cast to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img)

    ret, o1 = cv2.threshold(img_eq, min_th, max_th, cv2.THRESH_BINARY)

    # Erosion & Dilation
    kernel1 = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(o1, kernel1, iterations=2)
    img_dilation = cv2.dilate(img_erosion, kernel1, iterations=2)
    # white black color exchange()
    img_binary = 255 - img_dilation

    return img_binary
