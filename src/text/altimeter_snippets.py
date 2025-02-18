"""snippets for extracting altimeter data from images"""

# Library imports
import math
import numpy as np
from scipy import spatial
from shapely.geometry import LineString
from typing import Optional


def angle_between_lines(line1: tuple[float, float, float, float],
                        line2: tuple[float, float, float, float]) -> float:
    """
    Calculate the absolute angle between two lines in degrees.
    Args:
        line1 (tuple[float, float, float, float]): Coordinates of the first line as [x1, y1, x2, y2].
        line2 (tuple[float, float, float, float]): Coordinates of the second line as [x3, y3, x4, y4].
    Returns:
        angle (float): The angle between the two lines in degrees.
    """

    x1, y1 = line1[0], line1[1]
    x2, y2 = line1[2], line1[3]
    x3, y3 = line2[0], line2[1]
    x4, y4 = line2[2], line2[3]

    # handle vertical lines
    eps = 1e-8

    dx1 = (x2 - x1)
    dx2 = (x4 - x3)

    if abs(dx1) < eps:
        slope1= float('inf')
    else:
        slope1 = (y2 - y1) / (x2 - x1)

    if abs(dx2) < eps:
        slope2 = float('inf')
    else:
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


def check_middle_line_through_center(
    line_1: tuple[int, int, int, int],
    line_2: tuple[int, int, int, int],
    x_circle: float,
    y_circle: float,
    margin: float = 10
) -> tuple[bool, tuple[int, int, int, int]]:
    """
    Compute the angle bisector of the smaller angle formed by two lines (line_1, line_2).
    Then create a 'middle line' from the circle center in that bisector direction.
    Finally, check if the line is within some margin of the center.

    Args:
        line_1 (tuple[int, int, int, int]): First line [x1, y1, x2, y2].
        line_2 (tuple[int, int, int, int]): Second line [x3, y3, x4, y4].
        x_circle (float): X-coordinate of the circle center.
        y_circle (float): Y-coordinate of the circle center.
        margin (float): Max allowed distance from circle center to the bisector line.
                        (Often not strictly necessary if the line starts at the center.)

    Returns:
        (bool, (x1, y1, x2, y2)):
            A tuple where the first element indicates whether the middle line
            passes near the circle center (within `margin`), and the second is
            the 4-tuple of the bisecting line's coordinates.
    """

    # --- 1) Midpoints of each line ---
    x1, y1, x2, y2 = line_1
    x3, y3, x4, y4 = line_2

    mx1, my1 = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    mx2, my2 = (x3 + x4) / 2.0, (y3 + y4) / 2.0

    # --- 2) Direction vectors from center to each midpoint ---
    v1 = np.array([mx1 - x_circle, my1 - y_circle], dtype=float)
    v2 = np.array([mx2 - x_circle, my2 - y_circle], dtype=float)

    # If either midpoint is too close to center, angle is ill-defined
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-9 or norm_v2 < 1e-9:
        # Return False, with some dummy line or original
        return False, (int(x_circle), int(y_circle), int(x_circle), int(y_circle))

    # Normalize
    v1_u = v1 / norm_v1
    v2_u = v2 / norm_v2

    # --- 3) Angle between v1 and v2 ---
    dot = np.dot(v1_u, v2_u)
    # Clamp to avoid floating precision errors
    dot = max(-1.0, min(1.0, dot))
    angle = math.acos(dot)  # Radians

    # If angle > 180° (π radians), the smaller angle is 2π - angle.
    # We can pick the angle bisector accordingly:
    if angle <= math.pi:
        bisector = v1_u + v2_u  # smaller angle direct sum
    else:
        # They are >180°, so the "smaller angle" side is effectively the difference
        bisector = v1_u - v2_u

    # If the bisector is near zero, lines are nearly opposite
    bisector_norm = np.linalg.norm(bisector)
    if bisector_norm < 1e-9:
        # No well-defined bisector
        return False, (int(x_circle), int(y_circle), int(x_circle), int(y_circle))
    bisector_u = bisector / bisector_norm

    # --- 4) Create the 'middle line' from center outward in bisector direction ---
    # We'll choose a length about the max of the two line distances from center
    length = max(norm_v1, norm_v2)
    x_end = x_circle + bisector_u[0] * length
    y_end = y_circle + bisector_u[1] * length

    mid_line = (
        int(round(x_circle)), int(round(y_circle)),
        int(round(x_end)),    int(round(y_end))
    )

    # --- 5) Check distance from circle center to that line within 'margin' ---
    # Create a line from mid_line
    a = mid_line[1] - mid_line[3]
    b = mid_line[2] - mid_line[0]
    c = mid_line[0]*mid_line[3] - mid_line[2]*mid_line[1]

    denom = math.sqrt(a*a + b*b)
    if denom < 1e-9:
        # Degenerate line
        return False, mid_line

    # Perp distance from (x_circle, y_circle) to line
    dist_center = abs(a*x_circle + b*y_circle + c) / denom

    # If it passes near center within margin
    if dist_center <= margin:
        return True, mid_line
    else:
        return False, mid_line


def delete_short_lines(lines: list[tuple[int, int, int, int]],
                       threshold: int = 25) -> list[tuple[int, int, int, int]]:
    """
    Remove lines shorter than a specified threshold.

    Args:
        lines (list[tuple[int, int, int, int]]): List of lines where each line is
            represented by [x1, y1, x2, y2].
        threshold (int): Minimum length of the line to be kept.
    Returns:
        lines (list[tuple[int, int, int, int]]): List of lines after removing the short ones.
    """

    short_lines = []
    for line in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = line
        if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < threshold:
            short_lines.append(line)
    lines = [e for e in lines if e not in short_lines]

    return lines

def tip_points_outward(line1, line2, x_circle, y_circle):
    """
    Determine if the tip formed by line1 and line2 points away from the circle.

    The method is:
      1. Compute the intersection of line1 and line2.
      2. For each line, choose the endpoint that is farthest from the circle center.
      3. Check if the distance from the circle center to the intersection
         is greater than the distances from the circle center to both chosen endpoints.

    If yes, we say the tip is pointing in the good (outward) direction.
    """
    # Calculate intersection (tip) point.
    ix, iy = intersection_of_lines(line1, line2)
    if ix is None or iy is None:
        return False  # Lines do not intersect; cannot form a tip.

    # For a given line, pick the endpoint that is farther from the circle center.
    def far_endpoint(line):
        x1, y1, x2, y2 = line
        d1 = (x1 - x_circle) ** 2 + (y1 - y_circle) ** 2
        d2 = (x2 - x_circle) ** 2 + (y2 - y_circle) ** 2
        if d1 >= d2:
            return (x1, y1, d1)
        else:
            return (x2, y2, d2)

    ep1 = far_endpoint(line1)  # (x, y, squared_distance)
    ep2 = far_endpoint(line2)

    # Compute the squared distance from the circle center to the intersection.
    tip_dist_sq = (ix - x_circle) ** 2 + (iy - y_circle) ** 2

    # Check if the intersection is further away from the circle center than the far endpoints.
    if tip_dist_sq > ep1[2] and tip_dist_sq > ep2[2]:
        return True
    else:
        return False


def do_lines_form_tip(line_1: tuple[int, int, int, int],
                      line_2: tuple[int, int, int, int],
                      x_circle: float, y_circle: float, r_circle: float) -> bool:
    """
    Determine if two lines form a 'tip' around a circle defined by its center and radius.
    Args:
        line_1 (tuple[int, int, int, int]): First line coordinates [x1, y1, x2, y2].
        line_2 (tuple[int, int, int, int]): Second line coordinates [x3, y3, x4, y4].
        x_circle (float): X-coordinate of the circle center.
        y_circle (float): Y-coordinate of the circle center.
        r_circle (float): Radius of the circle.
    Returns:
        bool: True if lines form a tip according to specified conditions, False otherwise.
    """

    # 1. Get the intersection point of the two lines
    inter_x, inter_y = intersection_of_lines(line_1, line_2)
    if inter_x is None or inter_y is None:
        return False

    # 5. (Optional) Check that the lines are close enough to each other.
    ls1 = LineString([(line_1[0], line_1[1]), (line_1[2], line_1[3])])
    ls2 = LineString([(line_2[0], line_2[1]), (line_2[2], line_2[3])])
    if ls1.distance(ls2) > 30:
        return False

    # 2. Get the angles of each line (in radians)
    angle_1 = get_angle(line_1)
    angle_2 = get_angle(line_2)
    if angle_1 is None or angle_2 is None:
        return False

    # 3. Compute the acute angle between the lines (the tip angle)
    diff = abs(angle_1 - angle_2) % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    # For lines the “tip” is the smaller angle between them:
    tip_angle = diff if diff <= np.pi/2 else np.pi - diff

    # Use 40°–50° as acceptable tip angles (converted to radians)
    deg_min = np.deg2rad(30)  # ~0.524
    deg_max = np.deg2rad(50)  # ~0.873

    """
    import src.display.display_images as di
    img = np.full((586, 458), 255, dtype=int)
    img[0,0] = 0
    di.display_images(img, lines=[[line_1, line_2]])
    if not (deg_min < tip_angle < deg_max):
        return False
    """

    # 4. Check that the intersection is within an expected annulus around the center.
    # (r_circle/1.7 and r_circle/1.4 are heuristic inner and outer radii)
    dx = inter_x - x_circle
    dy = inter_y - y_circle
    dist_sq = dx**2 + dy**2
    r_inner = r_circle / 1.7
    r_outer = r_circle / 1.4
    if not (r_inner**2 < dist_sq < r_outer**2):
        return False

    if tip_points_outward(line_1, line_2, x_circle, y_circle) is False:
        return False

    return True


def get_angle(line: tuple[int, int, int, int]) -> Optional[float]:
    """
    Computes the angle of a line with respect to the vertical axis.

    Args:
        line (Tuple[int, int, int, int]): A tuple containing the coordinates of the line
                                                  in the format (x1, y1, x2, y2).

    Returns:
        Optional[float]: The angle in radians, or None if the angle could not be determined.
    """
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


def get_middle_line(lines: tuple[tuple[int, int, int, int],
                                 tuple[int, int, int, int]],
                    x_circle: float,
                    y_circle: float) -> tuple[tuple[int, int, int, int], float]:
    """
    Computes the middle line between two lines and the angle of this middle line.

    Args:
        lines (list[tuple[int, int, int, int]]): A list containing two lines,
            where each line is represented as a tuple of coordinates (x1, y1, x2, y2).
        x_circle (float): The x-coordinate of the circle's center.
        y_circle (float): The y-coordinate of the circle's center.

    Returns:
        tuple[tuple[int, int, int, int], float]: A tuple containing the middle
            line's coordinates as a tuple of points and the angle of the middle line in radians.
    """
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
        mid_line = (mid_p1[0], mid_p1[1], mid_p2[0], mid_p2[1])
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

        mid_line = (p1_x, p1_y, p2_x, p2_y)

    return mid_line, mid_angle_theta


def intersection_of_lines(line1: tuple[int, int, int, int],
                          line2: tuple[int, int, int, int]) -> \
        tuple[Optional[float], Optional[float]]:
    """
    Calculates the intersection point of two lines.

    Args:
        line1 (tuple[int, int, int, int]): Coordinates of the first line in the format (x1, y1, x2, y2).
        line2 (tuple[int, int, int, int]): Coordinates of the second line in the format (x3, y3, x4, y4).

    Returns:
        tuple[Optional[float], Optional[float]]: The intersection point (inter_x, inter_y).
                                                 Returns (None, None) if lines are parallel or identical.
    """

    # ch
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


def merge_lines(lines: list[tuple[int, int, int, int]],
                max_dist: int = 20, max_angle_diff: int = 1) -> list[tuple[int, int, int, int]]:
    """
    Merges similar lines into a single line based on pixel distance and angle difference.

    Args:
        lines (List[Tuple[float, float, float, float]]): A list of lines, where each line
            is represented as a tuple of coordinates (x1, y1, x2, y2).
        max_dist (int): The maximum distance between two lines to be considered similar.
        max_angle_diff (int): The maximum angle difference between two lines to be considered similar.
    Returns:
        List[Tuple[float, float, float, float]]: A list of merged lines.
    """

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
                if dist < max_dist and angle_diff < max_angle_diff:
                    pts = ([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    dist_mat = spatial.distance_matrix(pts, pts)
                    p1, p2 = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
                    x1_new, y1_new = pts[p1]
                    x2_new, y2_new = pts[p2]

                    lines[i] = (x1_new, y1_new, x2_new, y2_new)
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


def rearrange_line(line: tuple[float, float, float, float],
                   center_x: float, center_y: float) -> tuple[float, float, float, float]:
    """
    Rearranges the line such that the point closer to the center comes first.

    Args:
        line (tuple[float, float, float, float]): A tuple containing the coordinates of the line
            in the format (x1, y1, x2, y2).
        center_x (float): The x-coordinate of the center point.
        center_y (float): The y-coordinate of the center point.

    Returns:
        tuple[float, float, float, float]: A list of the rearranged line coordinates
            in the format [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = line
    dist_1 = math.dist([x1, y1], [center_x, center_y])
    dist_2 = math.dist([x2, y2], [center_x, center_y])
    if dist_1 > dist_2:
        line = (x2, y2, x1, y1)
    return line
