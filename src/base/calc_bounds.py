"""calculate bounds based on the transform and the shape of the image"""

import numpy as np
import rasterio

def calc_bounds(transform, shape):
    """
    Calculate the bounds of an image based on its transform and shape.

    :param transform: Affine transformation matrix (3x3 or 2x3).
    :param shape: Shape of the image (height, width).
    :return: Bounds of the image (min_x, min_y, max_x, max_y).
    """

    # convert to rasterio transform if numpy array
    if isinstance(transform, np.ndarray):

        if transform.shape == (9,):
            # reshape to 3x3
            transform = transform.reshape(3, 3)

        if transform.shape == (3, 3):
            # remove the last row
            transform = transform[:-1, :]

        transform = rasterio.transform.Affine(*transform.flatten())

    # get image height and width
    height, width = shape

    # Define the four corners of the image
    corners = np.array([
        [0, 0],
        [0, height],
        [width, 0],
        [width, height]
    ])


    # Apply the affine transformation to the corners
    transformed_corners = np.array([
        (transform * (x, y)) for x, y in corners
    ])

    # Extract the x and y coordinates
    x_coords = transformed_corners[:, 0]
    y_coords = transformed_corners[:, 1]

    # Calculate the bounds
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()

    return [min_x, min_y, max_x, max_y]
