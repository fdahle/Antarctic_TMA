"""Resamples tie points to absolute values"""

import numpy as np
from typing import Union


def resample_tie_points(points: Union[list[tuple[float, float]], np.ndarray],
                        trans_mat: np.ndarray) -> np.ndarray:
    """
    Resamples tie points to absolute values using a transformation matrix.
    Args:
        points (Union[List[Tuple[float, float]], np.ndarray]): A list of (x, y) coordinates
            or a numpy array of shape (n, 2).
        trans_mat (np.ndarray): A transformation matrix of shape (2, 3) or (3, 3).
    Returns:
        np.ndarray: A numpy array of transformed points of shape (n, 2).
    """

    # Ensure 'points' is a numpy array and convert to homogeneous coordinates
    points_np = np.atleast_2d(np.asarray(points))
    ones = np.ones((points_np.shape[0], 1))
    points_h = np.hstack([points_np, ones])

    # Check if transformation matrix is 2x3 and convert it to 3x3 for homogeneous multiplication
    if trans_mat.shape == (2, 3):
        trans_mat = np.vstack([trans_mat, [0, 0, 1]])  # Append row for homogeneous coordinates

    # Apply the transformation matrix
    points_tr = points_h.dot(trans_mat.T)  # Transpose to align for multiplication

    # Return only x and y, excluding the homogeneous coordinate
    return points_tr[:, :2].astype(int)
