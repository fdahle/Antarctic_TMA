# Package imports
import numpy as np
from math import atan2, degrees
from shapely import wkt

# Custom imports
import src.base.connect_to_database as ctd
import src.display.display_shapes as ds

# Debug settings
debug_show_points = False


def calc_azimuth(image_id, conn=None):
    # establish connection to psql
    if conn is None:
        conn = ctd.establish_connection()

    flight_path = image_id[2:6]

    # get the position of all images with the same flightpath
    sql_string = "SELECT image_id, ST_AsText(position_exact) AS position_exact FROM images_extracted " \
                 "WHERE SUBSTRING(image_id, 3, 4)='" + flight_path + "'"
    data = ctd.execute_sql(sql_string, conn)

    # remove the values where the position is empty
    data = data.dropna(subset=['position_exact'])

    # check if image_id is still in the data
    if image_id not in data['image_id'].values:
        return None

    # convert wkt to shapely points
    data['position_exact'] = data['position_exact'].apply(wkt.loads)

    if debug_show_points:
        ds.display_shapes(data['position_exact'].values)

    # Extract x and y coordinates for line fitting
    x = [point.x for point in data['position_exact']]
    y = [point.y for point in data['position_exact']]

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
