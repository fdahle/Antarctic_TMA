import numpy as np

import src.load.load_rema as lr


def get_elevation_for_points(points: np.ndarray, dem: np.ndarray | None = None,
                             zoom_level=10,
                             point_type="relative", dem_transform=None) -> np.ndarray:
    """
    Get the elevation for a list of points. The DEM can be absolute or relative. If
    the DEM is not provided, the function will download the DEM from the REMA dataset.
    Args:
        points (np-array):
        dem (str, optional): Digital Elevation Model to use. Defaults to None.
        zoom_level (int, optional): Zoom level of the DEM. Defaults to 10.
        point_type (str, optional): Type of points in the dem.
            Can be 'absolute' or 'relative'. Defaults to "relative".
    Returns:
        list: List of elevations.
    """

    # check for correct shapes of the arrays
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must be a 2D array with shape (n, 2)")
    if dem is not None and dem.ndim != 2:
        raise ValueError("DEM must be a 2D array")

    # load default dem if none is provided
    if dem is None:

        # get min/max x/y coordinates
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)

        # get the bounds and load the dem
        bounds = (min_x, min_y, max_x, max_y)
        dem, dem_transform = lr.load_rema(bounds, zoom_level=zoom_level,
                                          auto_download=True)

        if dem is None:
            raise ValueError("No DEM found")

    # assure we have a dem_transform
    if point_type == "absolute" and dem_transform is None:
        raise ValueError("A transform for the dem must be provided")

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
