import math

import numpy as np

from shapely.wkt import loads
from shapely.geometry import Point, LineString, Polygon
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

import base.connect_to_db as ctd
import base.print_v as p

import display.display_shapes as ds

_verbose = False

mode = "both"  # mode can be 'satellite', 'images' or 'both'
min_nr_of_images = 4

debug_show_line = False


def derive_image_position(image_id, mode="satellite", verbose=False, pbar=None):

    # get flight path from image
    flight_path = image_id[2:6]

    # get all images with this flight path
    sql_string = "SELECT image_id, SUBSTRING(image_id, 3, 4) AS flight_path, " \
                 "ST_AsText(footprint) AS footprint, " \
                 "ST_AsText(footprint_approx) as footprint_approx, " \
                 "ST_AsText(position_exact) AS position_exact, " \
                 "ST_AsText(position_approx) AS position_approx, " \
                 "position_error_vector, footprint_Type FROM images_extracted " \
                 f"WHERE SUBSTRING(image_id, 3, 4) ='{flight_path}'"
    data = ctd.get_data_from_db(sql_string)

    # condition is based on the mode
    if mode == "satellite":
        condition = (data['footprint'].notnull()) & \
                    (data['footprint_type'] == 'satellite')
    elif mode == "images":
        condition = (data['footprint'].notnull()) & \
                    (data['footprint_type'] == 'image')
    elif mode == "both":
        condition = data['footprint'].notnull()

    # create a subset with geo-referenced data
    georef_data = data[condition]

    # check how many images we have in our flight path
    nr_georef_images = georef_data.shape[0]

    # do we have enough images
    if nr_georef_images < min_nr_of_images:
        p.print_v("There are not enough geo-referenced images to derive image positions "
                  f"for flight {flight_path}", verbose=verbose, pbar=pbar)
        return None

    # order data-frame
    georef_data = georef_data.sort_values(by="image_id")

    # get all ids from flight-path as a list and convert list to integers
    id_list = georef_data["image_id"].tolist()
    id_list = [item[-4:] for item in id_list]
    id_list_int = [int(s) for s in id_list]

    # get all exact positions as a list and convert WKT strings to Shapely Point objects
    wkt_point_list = georef_data['position_exact'].tolist()
    wkt_footprint_list = georef_data['footprint'].tolist()
    point_list = [loads(wkt) for wkt in wkt_point_list]
    footprint_list = [loads(wkt) for wkt in wkt_footprint_list]

    # Extract x and y coordinates using list comprehension
    x = np.array([point.x for point in point_list])
    y = np.array([point.y for point in point_list])

    # standardize
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(x[..., None])
    y_train = y_scaler.fit_transform(y[..., None])

    # fit model
    model = HuberRegressor(epsilon=1)
    model.fit(x_train, y_train.ravel())

    regression_line_x = np.linspace(min(x), max(x), 100)

    # Transform these x-values to the standardized space
    regression_line_x_standardized = x_scaler.transform(regression_line_x[..., None])

    # Get predictions from the model for these x-values
    regression_line_y_standardized = model.predict(regression_line_x_standardized)

    # Transform predictions back to original space
    regression_line_y = y_scaler.inverse_transform(regression_line_y_standardized.reshape(-1, 1)).ravel()

    # get slope of regression line
    slope_standardized = model.coef_[0]
    slope = slope_standardized * (y_scaler.scale_ / x_scaler.scale_)

    # get intercept of regression line
    intercept = y_scaler.mean_ - (slope * x_scaler.mean_)

    # Predicted values based on the regression line
    predicted_y = slope * x + intercept

    # Compute residuals
    residuals = y - predicted_y

    # Compute outlier threshold based on IQR
    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify outliers
    outliers = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]

    # convert to line
    regression_line = LineString(zip(regression_line_x, regression_line_y))

    # show outliers if wished
    debug_show_outliers = False
    if debug_show_outliers:

        # Create a copy of the original list to keep it intact
        point_list2 = point_list.copy()
        id_list2 = id_list.copy()

        # Sort the indices in reverse order to avoid index shifting issues
        sorted_outliers = sorted(outliers, reverse=True)

        # Initialize a list to store the popped elements
        outlier_list = []
        outlier_id_list = []

        # Pop elements from the list based on the sorted indices
        for index in sorted_outliers:
            if 0 <= index < len(point_list2):
                outlier_list.append(point_list2.pop(index))
                outlier_id_list.append(id_list2.pop(index))

        ds.display_shapes([point_list2, outlier_list, regression_line],
                          subtitles=[id_list2, outlier_id_list, ""],
                          colors=["green", "red", "lightgray"],
                          title=f"Outliers for {flight_path}")

    # remove outliers from list
    x_cleaned = [item for i, item in enumerate(x) if i not in outliers]
    y_cleaned = [item for i, item in enumerate(y) if i not in outliers]
    id_list_int_cleaned = [item for i, item in enumerate(id_list_int) if i not in outliers]
    point_list_cleaned = [item for i, item in enumerate(point_list) if i not in outliers]
    footprint_list_cleaned = [item for i, item in enumerate(footprint_list) if i not in outliers]

    if len(x_cleaned) < 2:
        p.print_v("There are not enough geo-referenced images to derive image positions "
                  "(after outlier filtering) for flight {flight_path}", verbose=verbose, pbar=pbar)
        return None


    # function to find the angle of the regression line
    def find_angle_with_vertical(linestring):
        # Extract the coordinates of the two endpoints of the LineString
        x1, y1 = linestring.coords[0]
        x2, y2 = linestring.coords[-1]

        def find_angle(x1, y1, x2, y2):
            # Handle the vertical line case
            if x2 == x1:
                return 90  # or -90, depending on convention

            # Calculate the slope
            m = (y2 - y1) / (x2 - x1)

            # Calculate the angle in radians
            theta_radians = math.atan(m)

            # Convert to degrees
            theta_degrees = math.degrees(theta_radians)

            # Adjust for quadrant if necessary
            if x2 < x1:
                theta_degrees += 180

            # Keep the angle in the range [0, 360]
            theta_degrees = theta_degrees % 360

            return theta_degrees

        # Calculate the angle with the x-axis
        angle_x_axis = find_angle(x1, y1, x2, y2)

        # Calculate the angle with the y-axis (vertical line)
        if angle_x_axis <= 90:
            angle_vertical = 90 - angle_x_axis
        else:
            angle_vertical = angle_x_axis - 90

        return angle_vertical

    # we need the angle to later rotate the footprints accordingly
    angle = find_angle_with_vertical(regression_line)

    # TODO: FIND OUT WHY THIS WORKS?
    angle = angle + 45

    # function to find the average polygon size
    def find_polygon_size(polygons):

        # Extract information about size and rotation from polygons
        distances_to_corners = []

        for polygon in polygons:
            centroid = polygon.centroid
            coords = list(polygon.exterior.coords)

            # Compute the distance from centroid to a corner (to determine size)
            corner = Point(coords[0])
            distance_to_corner = centroid.distance(corner)
            distances_to_corners.append(distance_to_corner)

        # Calculate average distance to corner (for size) and average rotation
        avg_distance_to_corner = np.mean(distances_to_corners)

        return avg_distance_to_corner

    polygon_size = find_polygon_size(footprint_list_cleaned)

    # Find gaps in the list
    gaps = [(id_list_int_cleaned[i], id_list_int_cleaned[i + 1]) for i in range(len(id_list_int_cleaned) - 1) if
            id_list_int_cleaned[i + 1] - id_list_int_cleaned[i] > 1]

    # here we will save the estimated values
    estimated_ids = []   # the ids of every estimated image
    estimated_positions = []  # the position in (x,y)
    estimated_points = []  # the position as shapely point
    estimated_footprints = []  # the approx_footprint of this position

    # iterate all gaps
    for gap in gaps:

        # identify the start and end of an gap
        start_x, start_y = x_cleaned[id_list_int_cleaned.index(gap[0])], y_cleaned[id_list_int_cleaned.index(gap[0])]
        end_x, end_y = x_cleaned[id_list_int_cleaned.index(gap[1])], y_cleaned[id_list_int_cleaned.index(gap[1])]
        num_points_between = gap[1] - gap[0] - 1
        total_distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        avg_distance_local = total_distance / (num_points_between + 1)

        # Direction vector between the two points
        direction_x = (end_x - start_x) / total_distance

        # Using the direction vector and local average distance to compute steps the estimated position
        x_step = direction_x * avg_distance_local
        for j in range(num_points_between):
            # calc x
            x_pos = start_x + x_step * (j + 1)
            x_standardized = x_scaler.transform(np.array(x_pos).reshape(-1, 1))

            # calc y
            y_standardized = np.array(model.predict(x_standardized))
            y_pos = y_scaler.inverse_transform(y_standardized.reshape(-1, 1)).ravel()[0]

            # add position and shapely point
            estimated_positions.append((x_pos, y_pos))
            estimated_points.append(Point(x_pos, y_pos))

        # for every gap add the id
        for i in range(gap[0] + 1, gap[1]):
            estimated_ids.append(i)

    for point in estimated_points:
        # Assuming point contains the derived position
        center_x, center_y = point.x, point.y

        # Construct the rotated square using the average size and rotation
        half_side_length = polygon_size / (2 * np.sqrt(2))
        half_diagonal = half_side_length * np.sqrt(2) * 2
        # print(half_diagonal)
        dx = half_diagonal * np.cos(np.radians(angle))
        dy = half_diagonal * np.sin(np.radians(angle))
        square = Polygon([
            (center_x + dx, center_y + dy),
            (center_x - dy, center_y + dx),
            (center_x - dx, center_y - dy),
            (center_x + dy, center_y - dx)
        ])
        estimated_footprints.append(square)

    debug_show_estimated = True
    if debug_show_estimated:
        ds.display_shapes([regression_line, footprint_list, estimated_footprints, point_list, estimated_points],
                          subtitles=[None, None, None, id_list, estimated_ids],
                          colors=["lightgray", "green", "lightgray", "green", "red"],
                          title=flight_path)

    # extract the estimated point coordinate from the point list
    idx = estimated_ids.index(int(image_id[-4:]))
    point = estimated_points[idx]
    footprint = estimated_footprints[idx]

    return point, footprint

if __name__ == "__main__":

    # derive_image_position("CA180132V0094")

    _sql_string = "SELECT SUBSTRING(image_id, 3, 4) AS flight_path FROM images_extracted"
    _data = ctd.get_data_from_db(_sql_string)
    lst = set(_data['flight_path'].values.tolist())

    for elem in lst:
        breakout = False
        for _i in range(10, 99):
            try:
                print("Test CA" + elem + "32V00" + str(_i))
                derive_image_position("CA" + elem + "32V00" + str(_i), verbose=True)
            except:
                continue
            finally:
                breakout = True
                break
        if breakout:
            continue

    exit()

    derive_image_position("CA182232V0068", verbose=True)

    # derive_image_positions(verbose=_verbose)
