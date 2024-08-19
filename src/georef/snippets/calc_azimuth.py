"""Calculates the azimuth of an image based on its flight path relative to north."""

# Library imports
import numpy as np
from math import atan2, degrees
from typing import Optional

# Local imports
import src.display.display_shapes as ds

# Debug settings
debug_show_points = False


def calc_azimuth(image_id: str, geoframe) -> Optional[float]:
    """
    This function calculates the azimuth of the image flight line by fitting a linear model to
    the geographic positions (extracted from a database) of all images with the same flight path identifier.
    Azimuth is measured in degrees from the north, clockwise.

    Args:
        image_id (str): The unique identifier for the image, used to determine the flight path and extract
            relevant positions.

    Returns:
        Optional[float]: The azimuth in degrees as a float, or None if the image_id is not found in the database.
    """

    flight_path = image_id[2:6]

    # filter geopandas dataframe by flight path
    filtered_data = geoframe[geoframe["image_id"].str[2:6] == flight_path]

    # skip if no data is found
    if filtered_data.empty or filtered_data.shape[0] < 2:
        return None

    if debug_show_points:
        ds.display_shapes(filtered_data['geometry'].values)

    # Extract x and y coordinates for line fitting
    x = [point.x for point in filtered_data['geometry']]
    y = [point.y for point in filtered_data['geometry']]

    # Fit a line to these centers using polyfit for a degree 1 polynomial (linear fit)
    line_fit = np.polyfit(x, y, 1)

    # Slope from the linear fit
    slope = line_fit[0]

    # Calculate azimuth in degrees from the north
    azimuth = degrees(atan2(1, slope))  # atan2(x,y) used here for atan(y/x)
    azimuth = 90 - azimuth  # Adjusting as azimuth is traditionally measured from the north

    # Normalize azimuth to be within 0-360 degrees
    azimuth = azimuth % 360
    if azimuth < 0:
        azimuth += 360  # Corrects for negative values to ensure the range is 0-360

    return round(azimuth, 2)
