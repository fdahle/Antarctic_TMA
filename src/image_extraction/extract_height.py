import copy

import cv2 as cv2
import dlib
import json
import math
import numpy as np
import os

from scipy import spatial
from shapely.geometry import LineString

import base.print_v as p

import display.display_images as di

debug_show_altimeter_subset = False
debug_show_circle_binary = False
debug_show_circle_enhanced = False
debug_show_lines = False
debug_show_center_lines = False
debug_show_tip_lines = False
debug_show_parallel_lines = False
debug_show_pointer = False

# debug_show_altimeter_binary = False

center_radius_size = 25
matching_confidence_value = 20000000


def extract_height(image, image_id,
                   altimeter_bbox, circle,
                   min_th=None, max_th=None,
                   catch=True, verbose=False, pbar=None):
    """
    extract_height(image, image_id, altimeter_bbox, circle, min_th, max_th, catch, verbose, pbar):
    This function tries to convert the information on the altimeter into height information using
    computer-vision methods.
    Args:
        image:
        image_id:
        altimeter_bbox:
        circle:
        catch:
        verbose:
        pbar:

    Returns:

    """

    p.print_v(f"Start: extract_height ({image_id})", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if min_th is None:
        min_th = json_data["extract_height_min_th"]

    if max_th is None:
        max_th = json_data["extract_height_max_th"]

        # function calculate the angle of a line -> called by multiple functions

    def get_angle(_line):
        x1, y1, x2, y2 = _line

        _angle = None

        if x1 == x2:
            if y2 > y1:
                _angle = np.pi
            else:
                _angle = 0
        elif x2 > x1:
            _angle = math.atan((y1 - y2) / (x2 - x1))
            _angle = np.pi / 2 - _angle
        elif x2 < x1:
            _angle = math.atan((y1 - y2) / (x2 - x1))
            _angle = np.pi * 3 / 2 - _angle

        return _angle

        # function to get the (potential) intersection of lines --> called by multiple functions

    def intersection_of_lines(_line1, _line2):
        x1, y1, x2, y2 = _line1
        x3, y3, x4, y4 = _line2

        if x1 == x2 and x3 != x4:
            k2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - k2 * x3
            inter_x = x1
            inter_y = k2 * inter_x + b2
        elif x1 != x2 and x3 == x4:
            k1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - k1 * x1
            inter_x = x3
            inter_y = k1 * inter_x + b1
        elif x1 != x2 and x3 != x4:
            k1, k2 = (y2 - y1) / (x2 - x1), (y4 - y3) / (x4 - x3)
            b1, b2 = y1 - k1 * x1, y3 - k2 * x3
            if k1 != k2:
                inter_x = (b2 - b1) / (k1 - k2)
                inter_y = k1 * inter_x + b1
            else:
                inter_x = None
                inter_y = None
        else:
            inter_x = None
            inter_y = None

        return inter_x, inter_y

    # copy the image to not change the original
    image = copy.deepcopy(image)

    # get the altimeter subset
    altimeter_subset = image[altimeter_bbox[2]:altimeter_bbox[3],
                             altimeter_bbox[0]:altimeter_bbox[1]]

    # make circle again relative
    circle = list(circle)
    circle[0] = circle[0] - altimeter_bbox[0]
    circle[1] = circle[1] - altimeter_bbox[2]
    circle = tuple(circle)

    # function to enhance the circle
    def enhance_circle(_img_circle, _circle):
        img_mask = np.zeros_like(_img_circle)
        img_mask = cv2.circle(img_mask, (_circle[0], _circle[1]), int(_circle[2] /
                                                                      1.55), 255, -1)

        # store locations with value 255 (the clock part)
        loc = np.where(img_mask == 255)
        # pick intensity values in these locations from the grayscale image:
        values = _img_circle[loc[0], loc[1]]

        # Histogram equalization
        eq_hist = cv2.equalizeHist(values)

        gray2 = np.zeros_like(_img_circle)
        for i, coord in enumerate(zip(loc[0], loc[1])):
            gray2[coord[0], coord[1]] = eq_hist[i][0]

        # sharpening
        img_gb = cv2.GaussianBlur(gray2, (0, 0), 3)
        img_sharpened = cv2.addWeighted(gray2, 2.0, img_gb, -1.0, 0)

        # de-noising & contrast enhancement on then eqHist image
        # img_inh = contrast_enhance(" eqHist_sharpen_tmp .jpg")
        # os.remove('eqHist_sharpen_tmp .jpg ')
        # return img_inh

        return img_sharpened

    # enhance the circle
    #circle_enhanced = enhance_circle(altimeter_subset, circle)

    # show the enhanced circle if wished
    #if debug_show_circle_enhanced:
    #    di.display_images([altimeter_subset, circle_enhanced], title="circle enhanced")

    # function to find lines in the circle
    def find_lines(_circle_enhanced, _min_th, _max_th):

        # function to make an image binary
        def image_to_binary(__img, __min_th=0, __max_th=255):
            # converse the image to binary
            # thresholding
            ret, o1 = cv2.threshold(__img, __min_th, __max_th, cv2.THRESH_BINARY)
            # Erosion & Dilation
            kernel1 = np.ones((3, 3), np.uint8)
            img_erosion = cv2.erode(o1, kernel1, iterations=2)
            img_dilation = cv2.dilate(img_erosion, kernel1, iterations=2)
            # white black color exchange()
            img_binary = 255 - img_dilation

            return img_binary

        # make image binary again
        circle_binary = image_to_binary(_circle_enhanced, _min_th, _max_th)

        # get the edge of the binary image using Canny
        _edges = cv2.Canny(image=circle_binary, threshold1=100, threshold2=200, apertureSize=5)

        # HoughLinesP
        _lines = cv2.HoughLinesP(
            _edges,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi / 180,  # Angle resolution in radians
            threshold=28,  # Min number of votes for valid line
            minLineLength=16,  # Min allowed length of line
            maxLineGap=5  # Max allowed gap between line for joining them
        )

        # if we haven't found lines we need to return None
        if _lines is None:
            return None

        # Iterate over lines for displaying
        _lines_list = []
        for points in _lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            _lines_list.append([x1, y1, x2, y2])

        return _lines_list

    # find the lines in the circle
    lines = find_lines(altimeter_subset, min_th, max_th)

    if debug_show_lines:
        di.display_images(altimeter_subset, title="All lines", lines=[lines])

    if lines is None or len(lines) < 2:
        p.print_v("Too few lines found", verbose, pbar=pbar)
        return None

    p.print_v(f"{len(lines)} lines are found", verbose, pbar=pbar)

    # check whether the middle line of two lines goes through a central area of a circle
    def check_middle_line_through_center(___line_1, ___line_2, ___x_circle, ___y_circle):

        def angleTrans(____theta):
            # theta is the angle between a line and y-axis up
            # delta is the angle between a line and x-axis right
            # this function transform theta to delta
            delta = ____theta - np.pi / 2
            if delta < 0:
                delta = delta + np.pi * 2
            # elif delta >
            return delta

        def get_middle_line(____lines, ____x_circle, ____y_circle):
            # lines_2 is a list contains two lines that has been paired
            # mid_angle_theta is the angle to compute reading
            # while mid_angle_delta is the angle to compute line slope etc.
            ____x1, ____y1, ____x2, ____y2 = ____lines[0]
            ____x3, ____y3, ____x4, ____y4 = ____lines[1]
            # get the angles of line1 and line2
            angle1, angle2 = get_angle(____lines[0]), get_angle(____lines[1])  # [0, 2*pi) from y-axis
            delta1, delta2 = angleTrans(angle1), angleTrans(angle2)
            if np.abs(angle1 - angle2) > np.pi * 3 / 2:
                mid_angle_theta = (angle1 + angle2) / 2 - np.pi
                if mid_angle_theta < 0:
                    mid_angle_theta = mid_angle_theta + 2 * np.pi
            else:
                mid_angle_theta = (angle1 + angle2) / 2

            if np.abs(angle1 - angle2) < 0.05:
                # parallel lines, which means the long hand
                mid_p1 = [(____x1 + ____x3) / 2, (____y1 + ____y3) / 2]
                mid_p2 = [(____x2 + ____x4) / 2, (____y2 + ____y4) / 2]
                ____mid_line = [mid_p1, mid_p2]
            else:
                # not parallel, which means the shorthand
                mid_angle_delta = (delta1 + delta2) / 2
                inter_p_x, inter_p_y = intersection_of_lines(____lines[0], ____lines[1])

                if mid_angle_delta == np.pi / 2 or mid_angle_delta == 3 * np.pi / 2:
                    p1_x = inter_p_x
                    p2_x = inter_p_x
                    p1_y = ____lines[np.array([np.abs(____y1 - ____y_circle),
                                               np.abs(____y3 - ____y_circle)]).argmin()][1]
                    p2_y = ____lines[np.array([np.abs(____y2 - ____y_circle),
                                               np.abs(____y4 - ____y_circle)]).argmax()][3]
                else:
                    p1_x = ____lines[np.array([np.abs(____x1 - ____x_circle),
                                               np.abs(____x3 - ____x_circle)]).argmin()][0]
                    p2_x = ____lines[np.array([np.abs(____x2 - ____x_circle),
                                               np.abs(____x4 - ____x_circle)]).argmax()][2]
                    k = math.tan(mid_angle_delta)
                    b = inter_p_y - k * inter_p_x
                    p1_y = k * p1_x + b
                    p2_y = k * p2_x + b

                ____mid_line = [[p1_x, p1_y], [p2_x, p2_y]]

            return ____mid_line, mid_angle_theta

        # get the middle line
        mid_line, mid_angle = get_middle_line((___line_1, ___line_2),
                                              ___x_circle, ___y_circle)

        # get coefficients of the line
        ___a = mid_line[0][1] - mid_line[1][1]
        ___b = mid_line[1][0] - mid_line[0][0]
        ___c = mid_line[0][0] * mid_line[1][1] - mid_line[1][0] * mid_line[0][1]

        if ___a == 0 and ___b == 0:
            return False
        d = np.abs(___a * ___x_circle + ___b * ___y_circle + ___c) / np.sqrt(___a ** 2 + ___b ** 2)
        if d < 10:
            return True
        else:
            return False

    # function to keep only the interesting lines
    def select_lines(_circle, _lines, _altimeter_subset):

        def delete_short_lines(__lines):
            short_lines = []
            for line in __lines:
                # Extracted points nested in the list
                x1, y1, x2, y2 = line
                if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < 25:
                    short_lines.append(line)
            __lines = [e for e in __lines if e not in short_lines]

            return __lines

        def do_lines_form_tip(__line_1, __line_2, __x_circle, __y_circle, __r_circle):

            inter_x, inter_y = intersection_of_lines(__line_1, __line_2)

            # lines do not intersect
            if inter_x is None or inter_y is None:
                return False

            r_inner = int(__r_circle / 1.7)
            r_outer = int(__r_circle / 1.4)

            angle_1 = get_angle(__line_1)
            angle_2 = get_angle(__line_2)

            if angle_1 is None or angle_2 is None:
                return False

            # compute angle_diff
            angle_diff = np.abs(angle_1 - angle_2)
            if angle_diff > np.pi * 3 / 2:
                angle_diff = np.pi * 2 - angle_diff

            # convert lines to linestring to calculate the distance between the lines
            x1, y1, x2, y2 = __line_1
            x3, y3, x4, y4 = __line_2

            # convert lines to linestring to check the distance between them
            ls1 = LineString([(x1, y1), (x2, y2)])
            ls2 = LineString([(x3, y3), (x4, y4)])
            dist = ls1.distance(ls2)

            # get XXXX
            xy = (inter_x - __x_circle) ** 2 + (inter_y - __y_circle) ** 2

            # get the middle line through center
            mid_line_through_center = check_middle_line_through_center(__line_1, __line_2,
                                                                       __x_circle, __y_circle)

            # final check if the lines form a tip or not
            if 0.9 > angle_diff > 0.60 and \
                    dist < 60 and r_inner ** 2 < xy < r_outer ** 2 and \
                    mid_line_through_center is True:
                return True
            else:
                return False

        # lines that are close by and have the same angle should be merged
        def merge_lines(__lines):

            while True:
                break_loop = False
                lines_merged = False
                for i, __line_1 in enumerate(__lines):
                    x1, y1, x2, y2 = __line_1
                    l1 = LineString([(x1, y1), (x2, y2)])

                    for j, __line_2 in enumerate(__lines):

                        if __line_1 == __line_2:
                            continue
                        x3, y3, x4, y4 = __line_2
                        l2 = LineString([(x3, y3), (x4, y4)])

                        # get distance and angle difference
                        dist = l1.distance(l2)
                        angle_diff = np.abs(get_angle(__line_1) - get_angle(__line_2))

                        # if lines are similar -> create new line
                        if dist < 20 and angle_diff < 1:
                            pts = ([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                            dist_mat = spatial.distance_matrix(pts, pts)
                            p1, p2 = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
                            x1_new, y1_new = pts[p1]
                            x2_new, y2_new = pts[p2]

                            __lines[i] = [x1_new, y1_new, x2_new, y2_new]
                            del __lines[j]
                            lines_merged = True
                            break_loop = True

                        if break_loop:
                            break
                    if break_loop:
                        break

                if lines_merged is False:
                    break

            return __lines

        # function to rearrange the lines to that the first point in the line is
        # always closer to the circle
        def rearrange_line(__line, __center_x, __center_y):
            x1, y1, x2, y2 = __line
            dist_1 = math.dist([x1, y1], [__center_x, __center_y])
            dist_2 = math.dist([x2, y2], [__center_x, __center_y])
            if dist_1 > dist_2:
                __line = [x2, y2, x1, y1]
            return __line

        # get center of circle
        x_circle = _circle[0]
        y_circle = _circle[1]

        # we define a new radius that only covers the center
        r_circle = center_radius_size

        # only keep lines that are going through the center of the circle
        selected_lines_center = []
        selected_lines_non_center = []
        for line in _lines:

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

        # delete the lines that are too short (<25)
        selected_lines_center = delete_short_lines(selected_lines_center)
        selected_lines_non_center = delete_short_lines(selected_lines_non_center)

        # merge lines together
        selected_lines_center = merge_lines(selected_lines_center)
        #selected_lines_non_center = merge_lines(selected_lines_non_center)

        if debug_show_center_lines:
            di.display_images([_altimeter_subset, _altimeter_subset],
                              title="Center & non-center lines",
                              lines=[selected_lines_center, selected_lines_non_center])

        # look for lines that are making the tip of the short pointer
        selected_lines_tip = []
        selected_lines_tip_debug = []  # we need an extra list for showing the center lines
        for line_1 in selected_lines_non_center:

            # rearrange line so that the center point is the first point
            line_1 = rearrange_line(line_1, x_circle, y_circle)

            # don't use line again
            if line_1 in selected_lines_tip_debug:
                continue

            # we want to compare lines with other lines -> therefore second iteration
            for line_2 in selected_lines_non_center:

                # rearrange line so that the center point is the first point
                line_2 = rearrange_line(line_2, x_circle, y_circle)

                # we don't need to compare the same lines
                if line_1 == line_2:
                    continue

                # don't use line again
                if line_1 in selected_lines_tip_debug or line_2 in selected_lines_tip_debug:
                    continue

                # check if lines form a tip
                is_tip = do_lines_form_tip(line_1, line_2, x_circle, y_circle, _circle[2])

                # save the lines if we have a tip
                if is_tip:
                    selected_lines_tip.append([line_1, line_2])
                    selected_lines_tip_debug.append(line_1)
                    selected_lines_tip_debug.append(line_2)

        p.print_v(f"{len(selected_lines_tip)} pair of lines are forming a tip", verbose, pbar=pbar)

        if len(selected_lines_tip) > 0 and debug_show_tip_lines:
            di.display_images(altimeter_subset, title="Tip lines", lines=[selected_lines_tip_debug])

        return selected_lines_center, selected_lines_tip

    # select from the lines the ones that are parallel or form a tip
    center_lines, tip_lines = select_lines(circle, lines, altimeter_subset)

    # we need to have center lines
    if len(center_lines) == 0:
        p.print_v("No center lines found", verbose, pbar=pbar)
        return None

    # we need to also have tip lines
    if len(tip_lines) == 0:
        p.print_v("No tip lines found", verbose, pbar=pbar)
        return None

    # function to create line pairs of lines
    def pair_lines(_lines_center):

        def angle_between_lines(line1, line2):
            x1, y1 = line1[0], line1[1]
            x2, y2 = line1[2], line1[3]
            x3, y3 = line2[0], line2[1]
            x4, y4 = line2[2], line2[3]

            # calc slope
            slope1 = (y2 - y1) / (x2 - x1)
            slope2 = (y4 - y3) / (x4 - x3)

            angle = math.atan2(slope2 - slope1, 1 + slope1 * slope2)

            return abs(math.degrees(angle))

        _parallel_lines = []

        for line1 in _lines_center:

            for line2 in _lines_center:

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

                an = angle_between_lines(line1, line2)

                if an < 5:

                    ls1 = LineString([(line1[0], line1[1]), (line1[2], line1[3])])
                    ls2 = LineString([(line2[0], line2[1]), (line2[2], line2[3])])
                    dist = ls1.distance(ls2)

                    if dist < 25:
                        _parallel_lines.append([line1, line2])

        return _parallel_lines

    # get the parallel lines
    parallel_lines = pair_lines(center_lines)

    # keep the longest parallel lines
    def filter_parallel_lines(para_lines):

        top_length = 0
        top_pair = None

        for para_line in para_lines:
            line1 = para_line[0]
            line2 = para_line[1]
            dist1 = math.sqrt((line1[2] - line1[0])**2 + (line1[3] - line1[1])**2)
            dist2 = math.sqrt((line2[2] - line2[0])**2 + (line2[3] - line2[1])**2)
            dist = np.mean([dist1, dist2])
            if dist > top_length:
                top_length = dist
                top_pair = para_line

        return top_pair

    # filter parallel lines
    correct_parallel_lines = filter_parallel_lines(parallel_lines)

    if debug_show_parallel_lines:
        di.display_images(altimeter_subset, title="Parallel lines",
                          lines=correct_parallel_lines)

    # function to calculate height from lines
    def lines2height(_lines_tip, _lines_parallel, _circle, _img):

        if len(_lines_tip) > 1:
            print("Too many lines tip found")
            return None

        if len(_lines_parallel) > 2:
            print("Too many parallel lines found")
            return None

        final_lines = []

        # get middle line of short
        inter_x, inter_y = intersection_of_lines(_lines_tip[0][0], _lines_tip[0][1])
        middle_line_short = [_circle[0], _circle[1], inter_x, inter_y]
        final_lines.append(middle_line_short)

        # get middle line of long
        mid_x = int((_lines_parallel[0][0] + _lines_parallel[1][0]) / 2)
        mid_y = int((_lines_parallel[0][1] + _lines_parallel[1][1]) / 2)
        middle_line_long = [_circle[0], _circle[1], mid_x, mid_y]
        final_lines.append(middle_line_long)

        # get height value for short pointer
        middle_angle_short = get_angle(middle_line_short)
        reading_short = int(10000 / (2 * np.pi) * middle_angle_short / 1000) * 1000  # only need to know how many thousands

        # get height value for long pointer
        middle_angle_long = get_angle(middle_line_long)
        reading_long = int(1000 / (2 * np.pi) * middle_angle_long / 100) * 100

        height = 10000 + reading_short + reading_long

        if debug_show_pointer:
            di.display_images(_img, lines=final_lines)

        return height

    # get height from line
    height = lines2height(tip_lines, correct_parallel_lines, circle, altimeter_subset)

    print(image_id, height)

    p.print_v(f"Finished: extract_height ({image_id})", verbose=verbose, pbar=pbar)

    return height

if __name__ == "__main__":
    import base.load_image_from_file as liff

    #img = liff.load_image_from_file("CA174332V0210", catch=False)
    #img = liff.load_image_from_file("CA180532V0022", catch=False)
    img = liff.load_image_from_file("CA180532V0023", catch=False)
    img = liff.load_image_from_file("CA180532V0024", catch=False)

    extract_height(img)
