# Package imports
import copy
import math
import numpy as np
from scipy import spatial
from shapely.geometry import LineString


def angle_between_lines(line1: list[float], line2: list[float]) -> float:
    """
    Calculate the absolute angle between two lines in degrees.
    Args:
        line1 (List[float]): Coordinates of the first line as [x1, y1, x2, y2].
        line2 (List[float]): Coordinates of the second line as [x3, y3, x4, y4].
    Returns:
        angle (float): The angle between the two lines in degrees.
    """

    x1, y1 = line1[0], line1[1]
    x2, y2 = line1[2], line1[3]
    x3, y3 = line2[0], line2[1]
    x4, y4 = line2[2], line2[3]

    # calc slope
    slope1 = (y2 - y1) / (x2 - x1)
    slope2 = (y4 - y3) / (x4 - x3)

    # get the angle
    angle = math.atan2(slope2 - slope1, 1 + slope1 * slope2)
    angle = abs(math.degrees(angle))

    return angle


def angle_trans(theta: float) -> float:
    """
    Transform an angle from y-axis up to x-axis right orientation.
    Args:
        theta (float): The angle from y-axis up.
    Returns:
        delta (float): The transformed angle in x-axis right orientation.
    """

    # theta is the angle between a line and y-axis up
    # delta is the angle between a line and x-axis right
    # this function transform theta to delta
    delta = theta - np.pi / 2
    if delta < 0:
        delta = delta + np.pi * 2

    return delta


def check_middle_line_through_center(line_1: list[float], line_2: list[float],
                                     x_circle: float, y_circle: float) -> bool:
    """
    Check if the middle line between two given lines passes through a specified circle center.
    Args:
        line_1 (List[float]): Coordinates of the first line [x1, y1, x2, y2].
        line_2 (List[float]): Coordinates of the second line [x3, y3, x4, y4].
        x_circle (float): X-coordinate of the circle's center.
        y_circle (float): Y-coordinate of the circle's center.
    Returns:
        bool: True if the middle line passes through the circle center, else False.
    """

    # get the middle line
    mid_line, mid_angle = get_middle_line((line_1, line_2),
                                          x_circle, y_circle)

    # get coefficients of the line
    a = mid_line[0][1] - mid_line[1][1]
    b = mid_line[1][0] - mid_line[0][0]
    c = mid_line[0][0] * mid_line[1][1] - mid_line[1][0] * mid_line[0][1]

    if a == 0 and b == 0:
        return False
    d = np.abs(a * x_circle + b * y_circle + c) / np.sqrt(a ** 2 + b ** 2)
    if d < 10:
        return True
    else:
        return False


def delete_short_lines(lines: list[list[float]], threshold: int = 25) -> list[list[float]]:
    """
    Remove lines shorter than a specified threshold.

    Args:
        lines (List[List[float]]): List of lines where each line is represented by [x1, y1, x2, y2].
        threshold (int): Minimum length of the line to be kept.
    Returns:
        lines (List[List[float]]): List of lines after removing the short ones.
    """

    short_lines = []
    for line in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = line
        if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < threshold:
            short_lines.append(line)
    lines = [e for e in lines if e not in short_lines]

    return lines


def do_lines_form_tip(line_1: list[float], line_2: list[float],
                      x_circle: float, y_circle: float, r_circle: float) -> bool:
    """
    Determine if two lines form a 'tip' around a circle defined by its center and radius.
    Args:
        line_1 (List[float]): First line coordinates [x1, y1, x2, y2].
        line_2 (List[float]): Second line coordinates [x3, y3, x4, y4].
        x_circle (float): X-coordinate of the circle center.
        y_circle (float): Y-coordinate of the circle center.
        r_circle (float): Radius of the circle.
    Returns:
        bool: True if lines form a tip according to specified conditions, False otherwise.
    """

    inter_x, inter_y = intersection_of_lines(line_1, line_2)

    # lines do not intersect
    if inter_x is None or inter_y is None:
        return False

    r_inner = int(r_circle / 1.7)
    r_outer = int(r_circle / 1.4)

    angle_1 = get_angle(line_1)
    angle_2 = get_angle(line_2)

    if angle_1 is None or angle_2 is None:
        return False

    # compute angle_diff
    angle_diff = np.abs(angle_1 - angle_2)
    if angle_diff > np.pi * 3 / 2:
        angle_diff = np.pi * 2 - angle_diff

    # convert lines to linestring to calculate the distance between the lines
    x1, y1, x2, y2 = line_1
    x3, y3, x4, y4 = line_2

    # convert lines to linestring to check the distance between them
    ls1 = LineString([(x1, y1), (x2, y2)])
    ls2 = LineString([(x3, y3), (x4, y4)])
    dist = ls1.distance(ls2)

    # get XXXX
    xy = (inter_x - x_circle) ** 2 + (inter_y - y_circle) ** 2

    # get the middle line through center
    mid_line_through_center = check_middle_line_through_center(line_1, line_2,
                                                               x_circle, y_circle)

    # final check if the lines form a tip or not
    if 0.95 > angle_diff > 0.60 and \
            dist < 60 and r_inner ** 2 < xy < r_outer ** 2 and \
            mid_line_through_center is True:
        return True
    else:
        return False


def get_angle(line):
    x1, y1, x2, y2 = line

    angle = None

    if x1 == x2:
        if y2 > y1:
            angle = np.pi
        else:
            angle = 0
    elif x2 > x1:
        angle = math.atan((y1 - y2) / (x2 - x1))
        angle = np.pi / 2 - angle
    elif x2 < x1:
        angle = math.atan((y1 - y2) / (x2 - x1))
        angle = np.pi * 3 / 2 - angle

    return angle


def get_middle_line(lines, x_circle, y_circle):
    # lines_2 is a list contains two lines that has been paired
    # mid_angle_theta is the angle to compute reading
    # while mid_angle_delta is the angle to compute line slope etc.
    x1, y1, x2, y2 = lines[0]
    x3, y3, x4, y4 = lines[1]
    # get the angles of line1 and line2
    angle1, angle2 = get_angle(lines[0]), get_angle(lines[1])  # [0, 2*pi) from y-axis
    delta1, delta2 = angle_trans(angle1), angle_trans(angle2)
    if np.abs(angle1 - angle2) > np.pi * 3 / 2:
        mid_angle_theta = (angle1 + angle2) / 2 - np.pi
        if mid_angle_theta < 0:
            mid_angle_theta = mid_angle_theta + 2 * np.pi
    else:
        mid_angle_theta = (angle1 + angle2) / 2

    if np.abs(angle1 - angle2) < 0.05:
        # parallel lines, which means the long hand
        mid_p1 = [(x1 + x3) / 2, (y1 + y3) / 2]
        mid_p2 = [(x2 + x4) / 2, (y2 + y4) / 2]
        mid_line = [mid_p1, mid_p2]
    else:
        # not parallel, which means the shorthand
        mid_angle_delta = (delta1 + delta2) / 2
        inter_p_x, inter_p_y = intersection_of_lines(lines[0], lines[1])

        if mid_angle_delta == np.pi / 2 or mid_angle_delta == 3 * np.pi / 2:
            p1_x = inter_p_x
            p2_x = inter_p_x
            p1_y = lines[np.array([np.abs(y1 - y_circle),
                                   np.abs(y3 - y_circle)]).argmin()][1]
            p2_y = lines[np.array([np.abs(y2 - y_circle),
                                   np.abs(y4 - y_circle)]).argmax()][3]
        else:
            p1_x = lines[np.array([np.abs(x1 - x_circle),
                                   np.abs(x3 - x_circle)]).argmin()][0]
            p2_x = lines[np.array([np.abs(x2 - x_circle),
                                   np.abs(x4 - x_circle)]).argmax()][2]
            k = math.tan(mid_angle_delta)
            b = inter_p_y - k * inter_p_x
            p1_y = k * p1_x + b
            p2_y = k * p2_x + b

        mid_line = [[p1_x, p1_y], [p2_x, p2_y]]

    return mid_line, mid_angle_theta


def intersection_of_lines(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

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


def merge_lines(lines):

    while True:
        break_loop = False
        lines_merged = False
        for i, line_1 in enumerate(lines):
            x1, y1, x2, y2 = line_1
            l1 = LineString([(x1, y1), (x2, y2)])

            for j, line_2 in enumerate(lines):

                if line_1 == line_2:
                    continue

                x3, y3, x4, y4 = line_2
                l2 = LineString([(x3, y3), (x4, y4)])

                # get distance and angle difference
                dist = l1.distance(l2)
                angle_diff = np.abs(get_angle(line_1) - get_angle(line_2))

                # if lines are similar -> create new line
                if dist < 20 and angle_diff < 1:
                    pts = ([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    dist_mat = spatial.distance_matrix(pts, pts)
                    p1, p2 = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
                    x1_new, y1_new = pts[p1]
                    x2_new, y2_new = pts[p2]

                    lines[i] = [x1_new, y1_new, x2_new, y2_new]
                    del lines[j]
                    lines_merged = True
                    break_loop = True

                if break_loop:
                    break
            if break_loop:
                break

        if lines_merged is False:
            break

    return lines


def rearrange_line(line, center_x, center_y):
    x1, y1, x2, y2 = line
    dist_1 = math.dist([x1, y1], [center_x, center_y])
    dist_2 = math.dist([x2, y2], [center_x, center_y])
    if dist_1 > dist_2:
        line = [x2, y2, x1, y1]
    return line
