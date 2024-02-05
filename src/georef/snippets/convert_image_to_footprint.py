import numpy as np
import rasterio.features
import shapely.ops

from affine import Affine
from numpy import ndarray
from shapely.geometry import Polygon, MultiPolygon


def convert_image_to_footprint(image: ndarray, transform: Affine, no_data_value: int = 0) -> Polygon:
    """
    Converts a raster image into a footprint polygon by creating a mask to identify
    valid data points, converting these points to polygons, and then simplifying and
    selecting the largest contiguous polygon as the footprint.

    Args:
        image (ndarray): The input raster image as a NumPy array.
        transform (Affine): The affine transform associated with the raster image,
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

    # create a mask of this image to check where's no data
    mask = np.ones_like(image)
    mask[image == no_data_value] = 0

    # convert the raster cell to polygons
    shapes = rasterio.features.shapes(mask, transform=transform)

    # merge all polygons
    shape = shapely.ops.unary_union([shapely.geometry.shape(shape) for shape, val in shapes if val == 1])

    # flatten the polygon lines
    polygon = shape.simplify(100)

    # Initialize the biggest polygon and its area
    final_poly = None
    max_area = 0

    if isinstance(polygon, Polygon):
        polygons = [polygon]  # Convert single polygon to a list of polygons
    elif isinstance(polygon, MultiPolygon):
        polygons = polygon.geoms  # Access the individual polygons within the MultiPolygon
    else:
        raise ValueError("The type of polygon is undefined")

    # Iterate over all polygons
    for poly in polygons:

        # Calculate the area of the current polygon
        area = poly.area

        # If the current polygon's area is larger than the current maximum
        if area > max_area:
            # Update the biggest polygon and the maximum area
            final_poly = poly
            max_area = area

    # Create a new polygon without any interior rings
    final_poly = Polygon(final_poly.exterior)

    return final_poly
