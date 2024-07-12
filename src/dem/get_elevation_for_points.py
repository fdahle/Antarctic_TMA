import numpy as np

import src.load.load_rema as lr

def get_elevation_for_points(points, dem=None, point_type="relative") -> np.ndarray:
    """Get the elevation for a list of points.

    Args:
        points (np-array):
        dem (str, optional): Digital Elevation Model to use. Defaults to None.
        default (str, optional): Default DEM to use if dem is None. Defaults to "REMA".

    Returns:
        list: List of elevations.
    """

    # load default dem if none is provided
    if dem is None:

        # get min/max x/y coordinates
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)

        # get the bounds and load the dem
        bounds = (min_x, min_y, max_x, max_y)
        dem, dem_transform = lr.load_rema(bounds, zoom_level=32)

        if dem is None:
            raise ValueError("No DEM found")

    # get the elevation for each point
    elevations = []
    for point in points:
        x, y = point

        # we need to convert the absolute coordinates to relative coordinates
        if point_type == 'absolute':
            x = int((x - dem_transform[2]) / dem_transform[0])  # dem_transform[0] is the pixel width
            y = int((y - dem_transform[5]) / dem_transform[4])  # dem_transform[4] is the pixel height

        elevations.append(dem[y, x])

    return np.asarray(elevations)