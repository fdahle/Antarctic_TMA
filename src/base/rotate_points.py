import numpy as np


def rotate_points(points, angle, rotated_center, origin_center):

    # no need to rotate points with 0 degree angle
    if angle % 360 == 0:
        return points

    # Ensure input is a NumPy array
    points = np.array(points)
    rotated_center = np.array(rotated_center)
    origin_center = np.array(origin_center)

    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)

    # Translate points to origin (rotation center)
    translated_points = points - rotated_center

    # Create the rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Rotate points
    rotated_points = np.dot(translated_points, rotation_matrix.T)

    # Translate points back and adjust for new center
    final_points = rotated_points + origin_center

    return final_points
