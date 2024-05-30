import numpy as np
import statistics

from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points
from shapely.wkt import loads
from typing import List, Union

import src.display.display_shapes as ds

debug_plot_shapes = True
debug_print_distance = True


def verify_image_position(footprint: Union[Polygon, str],
                          line_footprints: List[Union[Polygon, str]],
                          distance_percentage: int = 20) -> bool:
    """
    Verifies if the given footprint's center is within a specified distance from a line fitted through
    the centers of a list of line footprints. The footprints can be provided as either Shapely Polygons
    or WKT (Well-Known Text) strings.

    Args:
        footprint (Union[Polygon, str]): The shapely Polygon or WKT string representing the footprint
            to verify.
        line_footprints (List[Union[Polygon, str]]): A list of shapely Polygons or WKT strings
            representing line footprints.
        distance_percentage (float): The maximum allowed difference in percentage from the
            average size of the footprints.

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
    line_x = [point.x for point in line_centers]
    line_y = [point.y for point in line_centers]

    # Fit a line to these centers using polyfit for a degree 1 polynomial (linear fit)
    line_fit = np.polyfit(line_x, line_y, 1)

    # Create a line function from the fit
    line_func = np.poly1d(line_fit)

    min_x = min(min(line_x), footprint_center.x)
    max_x = max(max(line_x), footprint_center.x)

    # Generate a long line for distance calculation;
    x_long = np.linspace(min_x - 1, max_x + 1, num=2)
    y_long = line_func(x_long)

    # Create a shapely LineString from the fitted line
    fitted_line = LineString(zip(x_long, y_long))

    # get avg distance from the footprints to the fitted line
    distances = [l_footprint.centroid.distance(fitted_line) for l_footprint in line_footprints]
    avg_distance = sum(distances) / len(distances)

    # Calculate the global average length of the sides of the squares
    global_average = statistics.mean(
        [statistics.mean(
            [LineString([coords[i], coords[(i + 1) % len(coords[:-1])]]).length for i in range(len(coords) - 1)])
         for coords in [list(l_footprint.exterior.coords) for l_footprint in line_footprints]]
    )

    # Get the distance between the footprint's center and the fitted line
    distance = footprint_center.distance(fitted_line)

    if debug_print_distance:
        print(f"Distance from footprint to line: {distance}")

    rejected = distance >= (distance_percentage * global_average) / 100

    if debug_plot_shapes and rejected:
        nearest_point_on_line = nearest_points(footprint_center, fitted_line)[1]
        shortest_line = LineString([footprint_center, nearest_point_on_line])

        print(shortest_line.length, distance, global_average)

        style_config = {
            "title": "Avg. Distance: {:.2f}, Distance: {:.2f}, "
                     "Rejected: {}".format(avg_distance, distance, rejected),
            "colors": ["green", "red", "gray", "black"]
        }
        ds.display_shapes([footprint, line_footprints, fitted_line, shortest_line],
                          style_config=style_config)

    # Check if the distance is below the threshold
    return rejected
