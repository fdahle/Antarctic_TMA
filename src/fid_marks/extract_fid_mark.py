"""extract a fiducial mark from an image"""

# Library imports
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
                     display: bool = False,
                     debug_display: bool = False) -> Optional[tuple[int, int]]:
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

    if debug_display:
        di.display_images([subset])

    # blur (to make the binarize easier) and binarize the subset (based on otsu, so the binarize is based on
    # the values in the subset
    subset_blurred = cv2.GaussianBlur(subset, (5, 5), 0)
    _, subset_binarized = cv2.threshold(subset_blurred, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # apply canny edge detection
    subset_canny_edge = cv2.Canny(subset_binarized, 100, 100)

    if debug_display:
        di.display_images([subset_canny_edge])

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

    if debug_display:
        style_config = {"line_width": 2}
        di.display_images([subset_canny_edge], lines=[all_lines], style_config=style_config)

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

    if debug_display:
        style_config={"line_width": 2}
        di.display_images([subset_canny_edge], lines=[[mid_line]], style_config=style_config)

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

        if debug_display:
            di.display_images([fid_subset])

        # blur this subset
        fid_subset_blurred = cv2.GaussianBlur(fid_subset, (5, 5), 0)

        if debug_display:
            di.display_images([fid_subset_blurred])


        # --- Try Hough Circle Detection ---
        fid_mark_local = None
        circles = cv2.HoughCircles(fid_subset_blurred,
                                   cv2.HOUGH_GRADIENT,
                                   dp=1.2,
                                   minDist=20,
                                   param1=50,
                                   param2=15,
                                   minRadius=3,
                                   maxRadius=15)

        if circles is not None:
            # Take the circle closest to the center of the subset
            circles = np.squeeze(circles, axis=0)
            min_distance = float('inf')
            mid_x_local = fid_subset.shape[1] / 2
            mid_y_local = fid_subset.shape[0] / 2

            for x, y, r in circles:
                if key in ["n", "s"]:
                    distance = abs(x - mid_x_local)
                else:
                    distance = abs(y - mid_y_local)

                if distance < min_distance:
                    min_distance = distance
                    fid_mark_local = (x, y)

            if fid_mark_local is not None:
                # Convert local fid mark to absolute image coordinates
                fid_mark = (int(fid_mark_local[0] + min_x_sm), int(fid_mark_local[1] + min_y_sm))
                break  # Hough succeeded → no need to check moments

        # --- Fallback: use Moments if no circle found ---
        if fid_mark_local is None:
            # Threshold
            ind = np.bincount(fid_subset_blurred.flatten()).argmax()
            thresh_val = ind + 10
            ret, fid_th = cv2.threshold(fid_subset_blurred, thresh_val, 255, cv2.THRESH_BINARY)

            # Erode and dilate
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(fid_th, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=1)

            # Contours
            contours, hierarchy = cv2.findContours(dilated, 1, 2)

            all_marks = []
            for elem in contours:
                m = cv2.moments(elem)
                size = m["m00"]

                if size == 0:
                    continue

                c_x = int(m["m10"] / m["m00"])
                c_y = int(m["m01"] / m["m00"])

                if 8 < size < 100:
                    all_marks.append([c_x, c_y])

            if len(all_marks) > 0:
                mid_of_image_x = int(dilated.shape[1] / 2)
                mid_of_image_y = int(dilated.shape[0] / 2)

                min_distance = float('inf')
                fid_mark_local = None

                for mark in all_marks:
                    if key in ["n", "s"]:
                        distance = abs(mid_of_image_x - mark[0])
                    else:
                        distance = abs(mid_of_image_y - mark[1])

                    if distance < min_distance:
                        fid_mark_local = mark

                if fid_mark_local is not None:
                    fid_mark = (fid_mark_local[0] + min_x_sm, fid_mark_local[1] + min_y_sm)
                    break

    if debug_display:
        di.display_images([fid_subset], points=[[fid_mark_local]])

    if display and fid_mark is not None:
        fid_mark_subset_x = fid_mark[0] - min_x
        fid_mark_subset_y = fid_mark[1] - min_y
        di.display_images(subset, points=[[(fid_mark_subset_x, fid_mark_subset_y)]])

    return fid_mark

if __name__ == "__main__":

    img_id = "CA033631L0063"
    sub_direction = "n"
    display = True

    # load the image
    import src.load.load_image as li
    img = li.load_image(img_id)

    # get the subset bounds for the fid mark extraction
    import src.base.connect_to_database as ctd
    conn = ctd.establish_connection()
    sql_string = f"SELECT subset_width, subset_height, subset_{sub_direction}_x, subset_{sub_direction}_y FROM images_fid_points"
    sql_string += f" WHERE image_id = '{img_id}'"
    data = ctd.execute_sql(sql_string, conn)

    min_x = int(data.iloc[0]["subset_" + sub_direction + "_x"])
    min_y = int(data.iloc[0]["subset_" + sub_direction + "_y"])
    max_x = int(min_x + data.iloc[0]["subset_width"])
    max_y = int(min_y + data.iloc[0]["subset_height"])

    bounds = (min_x, min_y, max_x, max_y)
    mark = extract_fid_mark(img, 'n', bounds, display=display, debug_display=display)

    print(f"Extracted fiducial mark: {mark}")