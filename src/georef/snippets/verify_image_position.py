import numpy as np

from shapely.geometry import LineString, Polygon
from shapely.wkt import loads
from typing import List, Union

import src.display.display_shapes as ds

debug_plot_shapes = True
debug_print_distance = True

def verify_image_position(footprint: Union[Polygon, str],
                          line_footprints: List[Union[Polygon, str]],
                          distance_threshold: float) -> bool:
    """
    Verifies if the given footprint's center is within a specified distance from a line fitted through
    the centers of a list of line footprints. The footprints can be provided as either Shapely Polygons
    or WKT (Well-Known Text) strings.

    Args:
        footprint (Union[Polygon, str]): The shapely Polygon or WKT string representing the footprint
            to verify.
        line_footprints (List[Union[Polygon, str]]): A list of shapely Polygons or WKT strings
            representing line footprints.
        distance_threshold (float): The maximum allowed distance from the line to consider
            the footprint verified.

    Returns:
        bool: True if the footprint's center is within the distance_threshold from the line,
            False otherwise.
    """

    # Convert WKT to Polygon if necessary
    if isinstance(footprint, str):
        footprint = loads(footprint)

    # Convert any WKT in line_footprints to Polygon
    line_footprints = [loads(fp) if isinstance(fp, str) else fp for fp in line_footprints]

    # Get the centroid of the footprint
    footprint_center = footprint.centroid

    # Get the centroids of the line footprints
    line_centers = [line.centroid for line in line_footprints]

    # Extract x and y coordinates for fitting
    x = [point.x for point in line_centers]
    y = [point.y for point in line_centers]

    # Fit a line to these centers using polyfit for a degree 1 polynomial (linear fit)
    line_fit = np.polyfit(x, y, 1)

    # Create a line function from the fit
    line_func = np.poly1d(line_fit)

    # Generate a long line for distance calculation;
    x_long = np.linspace(min(x) - 1, max(x) + 1, num=2)
    y_long = line_func(x_long)

    # Create a shapely LineString from the fitted line
    fitted_line = LineString(zip(x_long, y_long))

    # get all distances from the footprints to the fitted line
    distances = [l_footprint.centroid.distance(fitted_line) for l_footprint in line_footprints]

    # get average distance
    avg_distance = sum(distances) / len(distances)

    # Get the distance between the footprint's center and the fitted line
    distance = footprint_center.distance(fitted_line)

    print(avg_distance, distance)

    if debug_print_distance:
        print(f"Distance from footprint to line: {distance}")

    style_config={
        "title": "Distance: {:.2f}".format(distance),
        "colors": ["green", "red", "gray"]
    }
    if debug_plot_shapes:
        ds.display_shapes([footprint, line_footprints, fitted_line],
                          style_config=style_config)

    # Check if the distance is below the threshold
    return distance <= distance_threshold
