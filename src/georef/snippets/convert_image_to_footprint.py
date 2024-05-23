import numpy as np
import rasterio.features
import shapely.ops

from numpy import ndarray
from rasterio.transform import Affine
from shapely.geometry import Polygon


def convert_image_to_footprint(image: ndarray, transform: ndarray, extend=0,
                               no_data_value: int = 0) -> Polygon:
    """
    Converts a raster image into a footprint polygon by creating a mask to identify
    valid data points, converting these points to polygons, and then simplifying and
    selecting the largest contiguous polygon as the footprint.

    Args:
        image (ndarray): The input raster image as a NumPy array.
        transform (ndarray): The affine transform associated with the raster image,
                            which converts pixel coordinates to spatial coordinates.
        no_data_value (int): The value in the raster that represents no data.
                             Pixels with this value are excluded from the footprint. Defaults to 0.

    Returns:
        Polygon: The largest contiguous polygon (footprint) derived from the input image,
                 excluding areas with no data.

    Raises:
        ValueError: If the resulting shape is neither a Polygon nor a MultiPolygon, indicating
                    an unexpected result from the polygon creation process.
    """

    # check if georef transform is a numpy array
    if isinstance(transform, np.ndarray):

        if transform.shape[0] == 3 and transform.shape[1] == 3:
            a, b, c = transform[0]
            d, e, f = transform[1]
        elif transform.shape[0] == 9:
            a, b, c, d, e, f, _, _, _ = transform
        else:
            raise ValueError("The transformation matrix is not of the correct shape")

        # Create the Rasterio affine transform
        transform = Affine(a, b, c, d, e, f)

    # Calculate the coordinates of the raster's corners
    height, width = image.shape  # Get the dimensions of the array
    top_left = transform * (0, 0)
    top_right = transform * (width, 0)
    bottom_right = transform * (width, height)
    bottom_left = transform * (0, height)

    # Step 2: Define the polygon's corners in counter-clockwise order
    corners = [top_left, top_right, bottom_right, bottom_left, top_left]

    # Step 3: Create a Shapely polygon
    polygon = Polygon(corners)

    return polygon
