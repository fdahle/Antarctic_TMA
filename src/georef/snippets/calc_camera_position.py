"""Calculate camera position from a polygon."""

from shapely.geometry import Polygon, Point


def calc_camera_position(polygon: Polygon, catch: bool = True) ->\
        Point | None:
    """
    Calculates the camera position in x, y as the centroid of a polygon.

    Args:
        polygon (Polygon): The input polygon to calculate the centroid.
        catch (bool, optional): Whether to catch exceptions. Defaults to True.
    Returns:
        Point: The centroid of the polygon as a Point object if successful, None otherwise.
    """

    try:
        # calculate the camera position (just by taking the centroid)
        centroid = polygon.centroid
        x, y = centroid.coords[0]

    except (Exception,) as e:
        if catch:
            return None
        else:
            raise e

    point = Point(x, y)

    return point
