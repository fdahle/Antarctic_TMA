import shapely
from shapely.geometry import Point


def calc_camera_position(polygon, catch=True):

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
