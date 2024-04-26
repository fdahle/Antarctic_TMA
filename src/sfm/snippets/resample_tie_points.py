import numpy as np


def resample_tie_points(points, trans_mat):
    # Ensure 'points' is a numpy array for matrix operations
    points_np = np.atleast_2d(np.asarray(points))

    # Initialize an array for transformed points
    points_tr = np.empty_like(points_np)

    points_tr[:, 0] = trans_mat[0][0] * points_np[:, 0] + \
                      trans_mat[0][1] * points_np[:, 1] + trans_mat[0][2]

    points_tr[:, 1] = trans_mat[1][0] * points_np[:, 0] + \
                      trans_mat[1][1] * points_np[:, 1] + trans_mat[1][2]

    return points_tr
