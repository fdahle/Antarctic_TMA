import numpy as np
import cv2

def rotate_points(points, rotation_matrix, invert=False):
    """
    Projects points from the rotated and expanded image back to the original image.

    Args:
        points (list of tuples or np.ndarray): Points to project back, specified as (x, y).
        rotation_matrix (np.ndarray): 2x3 affine rotation matrix used for the original rotation.
        original_shape (tuple): The shape (height, width) of the original image.
        rotated_shape (tuple): The shape (height, width) of the rotated and expanded image.

    Returns:
        np.ndarray: The projected points as a numpy array.
    """
    # Convert points to numpy array if not already
    points = np.array(points, dtype=np.float32)

    # Invert the rotation matrix
    if invert:
        matrix = cv2.invertAffineTransform(rotation_matrix)
    else:
        matrix = rotation_matrix

    # Apply the inverse rotation matrix
    projected_points = cv2.transform(np.array([points]), matrix)[0]

    return projected_points
