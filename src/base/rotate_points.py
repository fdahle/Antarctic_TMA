import numpy as np


def rotate_points(points, angle, original_center, new_center):

    if angle % 360 == 0:
        return points

    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)

    rotated_points = []

    for x, y in points:
        # Translate point to origin (rotation center)
        translated_point = (x - original_center[0], y - original_center[1])

        # Rotate point
        rotated_point_x = (translated_point[0] * np.cos(angle_rad)) - (translated_point[1] * np.sin(angle_rad))
        rotated_point_y = (translated_point[0] * np.sin(angle_rad)) + (translated_point[1] * np.cos(angle_rad))

        # Translate point back and adjust for new image center
        final_point_x = int(rotated_point_x + new_center[0])
        final_point_y = int(rotated_point_y + new_center[1])

        # Convert the rotated point back to tuple and append to the list
        rotated_points.append([final_point_x, final_point_y])

    return rotated_points
