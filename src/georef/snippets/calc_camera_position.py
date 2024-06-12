"""Calculate camera position from a polygon."""

from shapely.geometry import Polygon, Point
from typing import Union


def calc_camera_position(polygon: Polygon, catch: bool = True) -> Union[Point, tuple[None, None]]:
    """
    Calculates the camera position in x, y as the centroid of a polygon.

    Args:
        polygon (Polygon): The input polygon to calculate the centroid.
        catch (bool, optional): Whether to catch exceptions and return None, None in case of an error. Defaults to True.

    Returns:
        Union[Point, Tuple[None, None]]: The centroid of the polygon as a Point object if successful,
                                         otherwise returns (None, None) if catch is True and an error occurs.
    """
    try:
        # calculate the camera position (just by taking the centroid)
        centroid = polygon.centroid
        x, y = centroid.coords[0]

    except (Exception,) as e:
        if catch:
            return None, None
        else:
            raise e

    point = Point(x, y)

    return point
