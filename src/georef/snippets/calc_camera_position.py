"""Calculate camera position from a polygon."""

from shapely.geometry import Polygon, Point


def calc_camera_position(image_id: str,
                         polygon: Polygon,) ->\
        Point | None:
    """
    Calculates the camera position in x, y as the centroid of a polygon.

    Args:
        polygon (Polygon): The input polygon to calculate the centroid.
        catch (bool, optional): Whether to catch exceptions. Defaults to True.
    Returns:
        Point: The centroid of the polygon as a Point object if successful, None otherwise.
    """

    if "32V" not in image_id:
        raise ValueError("Only nadir images are supported.")

    centroid = polygon.centroid
    x, y = centroid.coords[0]
    point = Point(x, y)

    return point
