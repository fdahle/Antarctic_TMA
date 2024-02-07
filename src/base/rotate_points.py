import numpy as np
import cv2

from typing import Union, List, Tuple


def rotate_points(points: Union[List[Tuple[float, float]], np.ndarray],
                  rotation_matrix: np.ndarray,
                  invert: bool = False) -> np.ndarray:
    """Rotate points using a given rotation matrix, optionally inverting the rotation.

    This function applies a rotation (or its inverse) to a collection of points
    using a provided affine rotation matrix.

    Args:
        points: A list of tuples or a numpy array of points specified as (x, y).
        rotation_matrix: A 2x3 affine rotation matrix used for the original rotation.
        invert: A boolean flag to indicate whether to invert the rotation matrix before
                applying it. Defaults to False, applying the rotation as is.

    Returns:
        A numpy array of the rotated points.
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
