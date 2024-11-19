import numpy as np
import matplotlib.pyplot as plt
from affine import Affine


def display_georef(image_array, transform, points=None, crs=None):
    """
    Display a georeferenced image with real-world coordinates in Matplotlib.

    Parameters:
    - image_array (numpy.ndarray): 2D NumPy array representing the image.
    - transform (Affine or similar): Affine transformation for georeferencing.
      Should be either an Affine object or a 3x3 NumPy array.
    - points (list of tuples, optional): A list of points to overlay on the image.
      Each point should be a tuple of (x, y) in real-world coordinates.
    - crs (str, optional): Coordinate reference system (e.g., "EPSG:3031") for display.
    """

    # Handle multi-channel images
    if image_array.ndim == 3 and image_array.shape[0] in (3, 4):  # Channels-first format
        image_array = np.moveaxis(image_array, 0, -1)  # Convert to (H, W, C)

    # Convert transform to Affine if it's not already
    if isinstance(transform, np.ndarray):
        transform = Affine(*transform.flatten()[:6])
    elif not isinstance(transform, Affine):
        raise ValueError("Transform must be an Affine object or a 3x3 NumPy array.")

    # Adjust extent for flipped y-direction
    if transform.e < 0:  # y-axis flipped
        extent = (
            transform.c,
            transform.c + transform.a * image_array.shape[1],  # x_max
            transform.f + transform.e * image_array.shape[0],  # y_min
            transform.f  # y_max
        )
    else:  # Standard y-axis
        extent = (
            transform.c,
            transform.c + transform.a * image_array.shape[1],  # x_max
            transform.f,  # y_min
            transform.f + transform.e * image_array.shape[0]  # y_max
        )

    # Plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image_array, extent=extent, origin='upper')
    plt.xlabel(f"Longitude/Easting ({crs})" if crs else "Longitude/Easting")
    plt.ylabel(f"Latitude/Northing ({crs})" if crs else "Latitude/Northing")
    plt.title("Georeferenced Image")

    # Overlay points if provided
    if points is not None:
        # Extract x and y coordinates from points
        x_coords, y_coords = zip(*points)
        plt.scatter(x_coords, y_coords, color='red', marker='x', label='Points')
        plt.legend()

    plt.show()
