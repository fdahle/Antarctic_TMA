import math
import numpy as np
import warnings

from shapely.affinity import rotate, translate
from shapely.geometry import Point, LineString, Polygon
from shapely.wkt import loads
from sklearn.linear_model import TheilSenRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import base.connect_to_db as ctd
import base.print_v as p

import display.display_shapes as ds

_verbose = True

_mode = "satellite"  # mode can be 'satellite', 'images' or 'both'
_polygon_mode = "outline"  # polygon mode can be 'outline' or 'exact'

debug_display_flight_path = False  # if true the flight path with the line and the polygons is plotted
debug_show_polygons = True
debug_show_outliers = False  # if true, the outliers are displayed as well

debug_delta = 'all'  # can be all or a number; plot only so many images around it


def derive_image_position(image_id,
                          mode="satellite", polygon_mode="outline",
                          min_nr_of_images=3,
                          max_range = 3,
                          polynomial_order=1,
                          return_line=False,
                          catch=True, verbose=False, pbar=None):
    """
    derive_image_position(image_id, mode, polygon_mode, min_nr_of_images, polynomial_order, return_line,
                          display_footprint, catch, verbose, pbar)
    This function tries to derive a position (and footprint) for a provided image id: The valid images of
    the flight-path (defined by mode) are used to calculate a regression line. Furthermore, the average distance
    between valid images is taken into account. This also works when an image-id is not between valid images
    Args:
        image_id (String): The id of the image for which we want to derive a position
        mode (String, "satellite"): Which type of geo-referencing should be considered as valid.
            Can be 'satellite' or 'image'
        polygon_mode (String, "outline"): Either the footprints are averaged polygons ('exact', more
            detailed) or more an outline of the polygons ('outline', suited as approximation)
        min_nr_of_images (int, 4): How many images are minimum required for this flight-path in order that we
            derive image positions
        polynomial_order (int, 1): max degree of the polynomial features to state how much curved the regression line
            can be (1 is straight line, 2 is a curved line and so on)
        return_line (Boolean, false): If true, the regression line is returned as well
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        point (Shapely-Point): A shapely point describing the center of the footprint
        footprint (Shapely polygon): A shapely polygon describing the footprint
        regression_line (shapely line): The regression line of this flight-path. Optional

    """

    # get flight path from image
    flight_path = image_id[2:6]

    # get the id as integer
    id_int = int(image_id[-4:])

    # get all images with this flight path
    sql_string = "SELECT image_id, SUBSTRING(image_id, 3, 4) AS flight_path, " \
                 "ST_AsText(footprint_exact) AS footprint_exact, " \
                 "ST_AsText(footprint_approx) as footprint_approx, " \
                 "ST_AsText(position_exact) AS position_exact, " \
                 "ST_AsText(position_approx) AS position_approx, " \
                 "position_error_vector, footprint_Type FROM images_extracted " \
                 f"WHERE SUBSTRING(image_id, 3, 4) ='{flight_path}' AND " \
                 f"image_id LIKE '%V%' ORDER BY image_id"
    data = ctd.get_data_from_db(sql_string, catch=catch)

    # condition is based on the mode
    if mode == "satellite":
        condition = (data['footprint_exact'].notnull()) & \
                    data['footprint_type'].isin(['sat', 'sat_est'])
    elif mode == "images":
        condition = (data['footprint_exact'].notnull()) & \
                    (data['footprint_type'] == 'img')
    elif mode == "both":
        condition = data['footprint_exact'].notnull()
    else:
        raise ValueError("The specified mode is not available")

    # create a subset with geo-referenced data
    georef_data = data[condition]

    # check how many images we have in our flight path
    nr_georef_images = georef_data.shape[0]

    # do we have enough images
    if nr_georef_images < min_nr_of_images:
        p.print_v(f"There are not enough geo-referenced images ({nr_georef_images}) to derive image positions "
                  f"for flight {flight_path}", verbose=verbose, pbar=pbar)
        if return_line:
            return "not_enough_images", None, None
        else:
            return "not_enough_images", None

    # order data-frame
    georef_data = georef_data.sort_values(by="image_id")

    # get all ids from flight-path as a list and convert list to integers
    id_arr = georef_data["image_id"].to_numpy()
    id_list = id_arr.tolist()
    id_list = [item[-4:] for item in id_list]
    id_list_int = [int(s) for s in id_list]

    # get all exact positions as a list and convert WKT strings to Shapely Point objects
    wkt_point_list = georef_data['position_exact'].tolist()
    wkt_footprint_list = georef_data['footprint_exact'].tolist()
    point_list = [loads(wkt) for wkt in wkt_point_list]
    footprint_list = [loads(wkt) for wkt in wkt_footprint_list]

    # Extract x and y coordinates of the exact polygons using list comprehension
    x = np.array([point.x for point in point_list])
    y = np.array([point.y for point in point_list])

    # standardize
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(x[..., None])
    y_train = y_scaler.fit_transform(y[..., None])

    # fit model
    try:
        model = make_pipeline(PolynomialFeatures(polynomial_order), TheilSenRegressor(fit_intercept=False))
        model.fit(x_train, y_train.ravel())
    except (Exception,):
        if return_line:
            return None, None, None
        else:
            return None, None

    # code to create the regression line (for displaying)
    regression_line_x = np.linspace(min(x), max(x), 100)
    regression_line_x_standardized = x_scaler.transform(regression_line_x[..., None])
    regression_line_y_standardized = model.predict(regression_line_x_standardized)
    regression_line_y = y_scaler.inverse_transform(regression_line_y_standardized.reshape(-1, 1)).ravel()
    regression_line = LineString(zip(regression_line_x, regression_line_y))

    # Standardize the x values for prediction
    x_standardized = x_scaler.transform(x[..., None])

    # Predict using the standardized x values
    predicted_y_standardized = model.predict(x_standardized)

    # Inverse transform the predicted values to get them in original scale
    predicted_y = y_scaler.inverse_transform(predicted_y_standardized.reshape(-1, 1)).ravel()

    # Compute residuals of the exact polygons
    residuals = y - predicted_y

    # Compute outlier threshold based on IQR
    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify outliers based on residuals
    outliers = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]

    # remove outliers from lists
    cleaned_ids = [item for i, item in enumerate(id_list) if i not in outliers]
    cleaned_ids_int = [item for i, item in enumerate(id_list_int) if i not in outliers]
    cleaned_points = [item for i, item in enumerate(point_list) if i not in outliers]
    cleaned_footprints = [item for i, item in enumerate(footprint_list) if i not in outliers]

    # get outliers
    outlier_ids_int = [item for i, item in enumerate(id_list_int) if i in outliers]
    outlier_points = [item for i, item in enumerate(point_list) if i in outliers]
    outlier_footprints = [item for i, item in enumerate(footprint_list) if i in outliers]

    if len(cleaned_ids) < 2:
        p.print_v("There are not enough geo-referenced images to derive image positions "
                  "(after outlier filtering) for flight {flight_path}", verbose=verbose, pbar=pbar)
        if return_line:
            return "not_enough_images", None, None
        else:
            return "not_enough_images", None

    # filter based on distance between the points


    """
    # get starting point of regression line with minimum x
    min_x = min([point.x for point in outlier_points]) - 10000
    adjusted_x_value = np.min(min_x)
    adjusted_x_standardized = x_scaler.transform([[adjusted_x_value]])
    predicted_y_standardized = model.predict(adjusted_x_standardized)
    predicted_y = y_scaler.inverse_transform([predicted_y_standardized])[0][0]
    min_start_point = Point(min_x, predicted_y)

    # Combine lists into pairs
    combined = list(zip(cleaned_ids, cleaned_ids_int, cleaned_points, cleaned_footprints))

    # Calculate distances of points to start of regression line and sort
    start_point = Point(regression_line.coords[0])
    sorted_combined = sorted(combined, key=lambda item: min_start_point.distance(item[2]))

    # Extract sorted points and IDs
    sorted_ids = [pair[0] for pair in sorted_combined]
    sorted_ids_int = [pair[1] for pair in sorted_combined]
    sorted_points = [pair[2] for pair in sorted_combined]
    sorted_footprints = [pair[3] for pair in sorted_combined]

    print(cleaned_points)
    print(cleaned_ids)
    print(sorted_points)
    print(sorted_ids)
    """

    # get direction of flight
    x_coords = [point.x for point in cleaned_points]

    # get first and last id and x
    fi_id, la_id = cleaned_ids_int[0], cleaned_ids_int[-1]
    fi_x, la_x = x_coords[0], x_coords[-1]

    # check the direction of both ids and the positions
    if fi_id < la_id:
        if fi_x < la_x:
            direction = "id_lr_x_lr"
        elif fi_x > la_x:
            direction = "id_lr_x_rl"
    elif fi_id > la_id:
        if fi_x < la_x:
            direction = "id_rl_x_lr"
        elif fi_x > la_x:
            direction = "id_rl_x_rl"

    # function to find the angle of the regression line
    def find_angle_with_vertical(linestring):
        # Extract the coordinates of the two endpoints of the LineString
        x1, y1 = linestring.coords[0]
        x2, y2 = linestring.coords[-1]

        def find_angle(_x1, _y1, _x2, _y2):
            # Handle the vertical line case
            if _x2 == _x1:
                return 90  # or -90, depending on convention

            # Calculate the slope
            m = (_y2 - _y1) / (_x2 - _x1)

            # Calculate the angle in radians
            theta_radians = math.atan(m)

            # Convert to degrees
            theta_degrees = math.degrees(theta_radians)

            # Adjust for quadrant if necessary
            if _x2 < _x1:
                theta_degrees += 180

            # Keep the angle in the range [0, 360]
            theta_degrees = theta_degrees % 360

            return theta_degrees

        # Calculate the angle with the _x-axis
        angle_x_axis = find_angle(x1, y1, x2, y2)

        # Calculate the angle with the y-axis (vertical line)
        if angle_x_axis <= 90:
            angle_vertical = 90 - angle_x_axis
        else:
            angle_vertical = angle_x_axis - 90

        return angle_vertical

    # we need the angle of regression line to later rotate the footprints accordingly
    angle = find_angle_with_vertical(regression_line)

    # here we will save the estimated values
    estimated_ids_int = []  # the ids of every estimated image
    estimated_ids = []  # the ids of every estimated image
    estimated_points = []  # the position as shapely point
    estimated_footprints = []  # the approx_footprint of this position

    for id_int in range(first_nr, last_nr):

        img_id = "CA" + flight_path + "32V" + str(id_int).zfill(4)

        if id_int in cleaned_ids_int:
            continue

        # get closest ids and polygons
        smaller_ids = sorted([num for num in cleaned_ids_int if num < id_int], key=lambda n: abs(id_int - n))
        larger_ids = sorted([num for num in cleaned_ids_int if num > id_int], key=lambda n: abs(id_int - n))

        smaller_ids = smaller_ids[:max_range]
        larger_ids = larger_ids[:max_range]

        smaller_footprints = [cleaned_footprints[cleaned_ids_int.index(i)] for i in smaller_ids]
        larger_footprints = [cleaned_footprints[cleaned_ids_int.index(i)] for i in larger_ids]
        smaller_points = [cleaned_points[cleaned_ids_int.index(i)] for i in smaller_ids]
        larger_points = [cleaned_points[cleaned_ids_int.index(i)] for i in larger_ids]

        print(smaller_ids, larger_ids)

        # sort lists
        if smaller_ids:
            smaller_ids, smaller_footprints, smaller_points = zip(*sorted(zip(smaller_ids, smaller_footprints, smaller_points)))
            smaller_ids = list(smaller_ids)
            smaller_footprints = list(smaller_footprints)
            smaller_points = list(smaller_points)

        if larger_ids:
            larger_ids, larger_footprints, larger_points = zip(
                *sorted(zip(larger_ids, larger_footprints, larger_points)))
            larger_ids = list(larger_ids)
            larger_footprints = list(larger_footprints)
            larger_points = list(larger_points)

        from itertools import combinations

        # calculate the avg distance between one polygon
        def average_point_distance(_ids, _points):

            # Compute pairwise distances
            _distances = []
            for (_id1, _point1), (_id2, _point2) in combinations(zip(_ids, _points), 2):
                _distance = _point1.distance(_point2)

                # Adjust for gap in IDs
                _gap = abs(_id1 - _id2)
                if _gap > 0:
                    _distance /= _gap

                _distances.append(_distance)

            # Compute average
            return sum(_distances) / len(_distances) if _distances else 0

        # avg distance between polygons
        small_avg_distance = average_point_distance(smaller_ids, smaller_points)
        large_avg_distance = average_point_distance(larger_ids, larger_points)

        # get the distance of target to the closest polygon
        if direction.startswith("id_lr"):
            smaller_distance = abs(id_int - smaller_ids[-1]) if smaller_ids else 0
            larger_distance = abs(larger_ids[0] - id_int) if larger_ids else 0
        elif direction.startswith("id_rl"):
            smaller_distance = abs(id_int - smaller_ids[0]) if smaller_ids else 0
            larger_distance = abs(larger_ids[-1] - id_int) if larger_ids else 0

        # convert distances to weights
        smaller_weight = larger_distance / (smaller_distance + larger_distance)
        larger_weight = smaller_distance / (smaller_distance + larger_distance)

        # get a weight average distance
        if smaller_weight == 0:
            avg_distance = small_avg_distance
        elif larger_weight == 0:
            avg_distance = large_avg_distance
        else:
            # for when both larger and smaller ids only have 1 value
            if small_avg_distance == 0 and large_avg_distance == 0:
                avg_distance = abs(smaller_points[0].x - larger_points[0].x)/abs(int(smaller_ids[0]) - int(larger_ids[0]))
            else:
                avg_distance = small_avg_distance * smaller_weight + large_avg_distance * larger_weight

        assert avg_distance != 0, f"Distance cannot be 0 {avg_distance}!!"

        print(id_int, direction, len(smaller_ids), len(larger_ids))

        # get the new x position of the polygon based on the regression line
        # (dependent if we have smaller or larger values)
        if len(smaller_ids) > 0:
            if direction == "id_lr_x_lr":
                x_pos = smaller_points[-1].x + avg_distance * smaller_distance
            elif direction == "id_lr_x_rl":
                x_pos = smaller_points[-1].x - avg_distance * smaller_distance
            elif direction == "id_rl_x_lr":
                x_pos = smaller_points[0].x + avg_distance * smaller_distance
            elif direction == "id_rl_x_rl":
                x_pos = smaller_points[0].x - avg_distance * smaller_distance

        elif len(larger_ids) > 0:
            if direction == "id_lr_x_lr":
                x_pos = larger_points[0].x - avg_distance * larger_distance
            elif direction == "id_lr_x_rl":
                x_pos = larger_points[0].x + avg_distance * larger_distance
            elif direction == "id_rl_x_lr":
                x_pos = larger_points[-1].x - avg_distance * larger_distance
            elif direction == "id_rl_x_rl":
                x_pos = larger_points[-1].x + avg_distance * larger_distance

        else:
            raise ValueError("Both empty")

        x_standardized = x_scaler.transform(np.array(x_pos).reshape(-1, 1))

        # calc y
        y_standardized = np.array(model.predict(x_standardized))
        y_pos = y_scaler.inverse_transform(y_standardized.reshape(-1, 1)).ravel()[0]

        # calculate the average polygon
        def average_polygon(_poly1_lst, _poly2_lst, _weight1, _weight2, _angle):

            # function to translate polygon to origin
            def translate_to_origin(__polygon):
                centroid = __polygon.centroid
                translated_polygon = translate(__polygon, xoff=-centroid.x, yoff=-centroid.y)
                return translated_polygon

            # translate all polygons in both lists to origin
            _poly1_lst_translated = [translate_to_origin(poly) for poly in _poly1_lst]
            _poly2_lst_translated = [translate_to_origin(poly) for poly in _poly2_lst]

            def __simplify_to_square(__polygon):

                # Compute the convex hull of the polygon
                hull = __polygon.convex_hull

                # Initialize variables to store the smallest square and its corresponding rotation
                min_area = float('inf')
                best_square = None
                best_angle = 0

                # Iterate through each edge of the convex hull
                for i in range(len(hull.exterior.coords) - 1):
                    # Calculate the angle to rotate the polygon so the edge aligns with the x-axis
                    p1, p2 = hull.exterior.coords[i], hull.exterior.coords[i + 1]
                    _angle = -np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi

                    # Rotate the polygon
                    rotated_poly = rotate(__polygon, _angle, origin='centroid')

                    # Compute the bounding box for the rotated polygon
                    _min_x, _min_y, _max_x, _max_y = rotated_poly.bounds
                    side_length = max(_max_x - _min_x, _max_y - _min_y)
                    square = Polygon([(_min_x, _min_y), (_min_x + side_length, _min_y),
                                      (_min_x + side_length, _min_y + side_length), (_min_x, _min_y + side_length)])

                    # Check if this square is the smallest encountered so far
                    if square.area < min_area:
                        min_area = square.area
                        best_square = square
                        best_angle = _angle

                # Rotate the best square back to align with the hull edge's original orientation
                best_square = rotate(best_square, -best_angle, origin='centroid')

                return best_square, best_angle

            _poly1_lst_angles = []
            _poly1_lst_simple = []
            for poly in _poly1_lst_translated:
                simple_poly, angle = __simplify_to_square(poly)
                _poly1_lst_simple.append(simple_poly)
                _poly1_lst_angles.append(angle)

            _poly2_lst_angles = []
            _poly2_lst_simple = []
            for poly in _poly2_lst_translated:
                simple_poly, angle = __simplify_to_square(poly)
                _poly2_lst_simple.append(simple_poly)
                _poly2_lst_angles.append(angle)

            def __sort_vertices(polygon):

                def compute_angle(point, _centroid):
                    # Returns the angle (in radians) of a point with respect to a centroid
                    return math.atan2(point[1] - _centroid[1], point[0] - _centroid[0])

                # Sorts the vertices of a polygon based on their angle with respect to the centroid
                centroid = polygon.centroid.coords[0]
                sorted_coords = sorted(list(polygon.exterior.coords)[:-1], key=lambda _p: compute_angle(_p, centroid))
                sorted_coords.append(sorted_coords[0])  # close the polygon
                return sorted_coords

            _poly1_lst_sorted = [__sort_vertices(poly) for poly in _poly1_lst_simple]
            _poly2_lst_sorted = [__sort_vertices(poly) for poly in _poly2_lst_simple]

            def __average_distance_to_corners(__lst):
                __polygon = Polygon(__lst)
                __centroid = __polygon.centroid
                __distances = [__centroid.distance(Point(point)) for point in
                             __polygon.exterior.coords[:-1]]  # the last point is the same as the first in a polygon
                return np.nanmean(__distances)

            # get average distances of corners to the centroid
            _poly1_lst_sorted_corner_distances = [__average_distance_to_corners(lst) for lst in _poly1_lst_sorted]
            _poly2_lst_sorted_corner_distances = [__average_distance_to_corners(lst) for lst in _poly2_lst_sorted]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                poly_1_avg_distance = np.nanmean(_poly1_lst_sorted_corner_distances)
                poly_2_avg_distance = np.nanmean(_poly2_lst_sorted_corner_distances)

                avg_angle_1 = np.nanmean(_poly1_lst_angles)
                avg_angle_2 = np.nanmean(_poly2_lst_angles)

            # get a weight average distance
            if smaller_weight == 0:
                poly_size = poly_1_avg_distance
                _good_angle = avg_angle_1
            elif larger_weight == 0:
                poly_size = poly_2_avg_distance
                _good_angle = avg_angle_2
            else:
                poly_size = poly_1_avg_distance * smaller_weight + poly_2_avg_distance * larger_weight
                _good_angle = avg_angle_1 * smaller_weight + avg_angle_2 * larger_weight

            half_side_length = poly_size / (2 * np.sqrt(2))
            half_diagonal = half_side_length * np.sqrt(2) * 2

            dx = half_diagonal * np.cos(np.radians(_good_angle))
            dy = half_diagonal * np.sin(np.radians(_good_angle))
            poly = Polygon([
                (+ dx, + dy),
                (- dy, + dx),
                (- dx, - dy),
                (+ dy, - dx)
            ])

            rot_angle = 360 - _angle - _good_angle
            rot_angle = 200
            poly = rotate(poly, rot_angle, origin='centroid')

            return poly

            all_averaged_coords = []

            # get average coords
            if not _poly1_lst_sorted and not _poly2_lst_sorted:
                all_averaged_coords = []
            elif not _poly1_lst_sorted:
                all_averaged_coords = [coord for coords in _poly2_lst_sorted for coord in coords]
            elif not _poly2_lst_sorted:
                all_averaged_coords = [coord for coords in _poly1_lst_sorted for coord in coords]
            else:
                for sorted_coords1, sorted_coords2 in zip(_poly1_lst_sorted, _poly2_lst_sorted):
                    averaged_coords = []
                    for (x1, y1), (x2, y2) in zip(sorted_coords1, sorted_coords2):
                        avg_x = (x1 * _weight1 + x2 * _weight2)
                        avg_y = (y1 * _weight1 + y2 * _weight2)
                        averaged_coords.append((avg_x, avg_y))
                    all_averaged_coords.extend(averaged_coords)

            # Use the convex hull of the averaged coordinates to obtain a single average polygon
            avg_poly = Polygon(all_averaged_coords)

            avg_poly = __simplify_to_square(avg_poly)
            return avg_poly

        # calc average polygon
        avg_poly = average_polygon(smaller_footprints, larger_footprints,
                                   smaller_weight, larger_weight, angle)

        # move polygon to the right position
        dx = x_pos - avg_poly.centroid.x
        dy = y_pos - avg_poly.centroid.y

        estimated_point = Point(x_pos, y_pos)
        estimated_poly = translate(avg_poly, xoff=dx, yoff=dy)

        estimated_ids.append(img_id)
        estimated_ids_int.append(id_int)
        estimated_points.append(estimated_point)
        estimated_footprints.append(estimated_poly)

    # display the flight path and polygons
    if debug_display_flight_path:

        # check if we want to show outliers
        if debug_show_outliers is False:
            outlier_footprints = []
            outlier_points = []

        # we want to show all
        if debug_delta == "all":

            # we can just copy the lists
            display_regression_line = regression_line
            display_cleaned_ids = cleaned_ids_int
            display_cleaned_footprints = cleaned_footprints
            display_cleaned_points = cleaned_points
            display_estimated_ids = estimated_ids_int
            display_estimated_footprints = estimated_footprints
            display_estimated_points = estimated_points
            display_outlier_ids = outlier_ids_int
            display_outlier_footprints = outlier_footprints
            display_outlier_points = outlier_points

        # reduce the number of things we show
        else:

            assert isinstance(debug_delta, int), "debug_delta is not a number"

            # init regression line
            display_regression_line = regression_line

            # init empty lists
            display_cleaned_ids = []
            display_cleaned_footprints = []
            display_cleaned_points = []
            display_estimated_ids = []
            display_estimated_footprints = []
            display_estimated_points = []
            display_outlier_ids = []
            display_outlier_footprints = []
            display_outlier_points = []

            # which numbers are we allowed to show
            display_nrs = list(range(max(0, int(image_id[-4:]) - int(debug_delta)),
                                     int(image_id[-4:]) + (int(debug_delta)) + 1))

            # filter cleaned
            for idx, elem in enumerate(cleaned_ids_int):
                if elem in display_nrs:
                    display_cleaned_ids.append(cleaned_ids_int[idx])
                    display_cleaned_footprints.append(cleaned_footprints[idx])
                    display_cleaned_points.append(cleaned_points[idx])

            # filter estimated
            for idx, elem in enumerate(estimated_ids_int):
                if elem in display_nrs:
                    display_estimated_ids.append(estimated_ids_int[idx])
                    display_estimated_footprints.append(estimated_footprints[idx])
                    display_estimated_points.append(estimated_points[idx])

            # filter outlier
            for idx, elem in enumerate(outlier_ids_int):
                if elem in display_nrs:
                    display_outlier_ids.append(outlier_ids_int[idx])
                    display_outlier_footprints.append(outlier_footprints[idx])
                    display_outlier_points.append(outlier_points[idx])

            # get min and max _x from the points
            combined_points = display_cleaned_points + display_estimated_points + display_outlier_points
            x_values = [point.x for point in combined_points]
            min_x, max_x = min(x_values), max(x_values)

        if debug_show_polygons is False:
            display_cleaned_footprints = []
            display_estimated_footprints = []
            display_outlier_footprints = []

        ds.display_shapes([display_regression_line,
                           display_cleaned_footprints, display_cleaned_points,
                           display_estimated_footprints, display_estimated_points,
                           display_outlier_footprints, display_outlier_points],
                          subtitles=[None, display_cleaned_ids,
                                     None, display_estimated_ids,
                                     None, display_outlier_ids, None],
                          colors=["black", "green", "green", "lightgray", "lightgray", "red", "red"],
                          alphas=[1, 1, 1, 0.5, 0.5, 1, 1],
                          title=flight_path)

    print(cleaned_ids_int)
    print(estimated_ids_int)
    print(image_id[-4:])

    # extract the estimated point coordinate from the point list
    if int(image_id[-4:]) in cleaned_ids_int:
        idx = cleaned_ids_int.index(int(image_id[-4:]))
        point = cleaned_points[idx]
        footprint = cleaned_footprints[idx]

    elif int(image_id[-4:]) in estimated_ids_int:
        idx = estimated_ids_int.index(int(image_id[-4:]))
        point = estimated_points[idx]
        footprint = estimated_footprints[idx]

    else:
        raise ValueError("Id not found")

    if return_line:
        return point, footprint, regression_line
    else:
        return point, footprint


if __name__ == "__main__":

    debug_display_flight_path = True
    debug_show_outliers = False
    debug_show_polygons = True

    flight_paths = [
        1962, 1849, 1846, 2163, 1719, 1817, 1827, 1844, 1803, 1644, 1825, 1833, 2121, 1832, 1816, 1834, 1845, 1829,
        2134, 1804, 1963, 1969, 1742, 2140, 1965, 1826, 1801, 2133, 1745, 1814, 2135, 2040, 1876, 2167, 1741, 1822,
        1724, 1744, 1847, 2139, 1942, 1893, 1721, 5127, 5120, 1982, 1720, 2041, 5124, 2042, 1898, 2039, 2043, 5126,
        5125, 1725, 1723, 1994, 1802, 1743, 1810, 1823, 1830, 1815, 1807, 1843]


    #flight_paths = [2121]

    for _flight_path in flight_paths:

        _sql_string = f"SELECT image_id FROM IMAGES WHERE SUBSTRING(image_id, 3, 4) ='{_flight_path}' AND " \
                      f"image_id LIKE '%V%' ORDER BY image_id"
        _data = ctd.get_data_from_db(_sql_string, catch=False)

        print(f"There are {_data.shape[0]} images for this flight path")

        random_id = _data.sample(1)['image_id'].iloc[0]
        first_nr = int(_data['image_id'].iloc[0][-4:])
        last_nr = int(_data['image_id'].iloc[-1][-4:])
        # random_id = "CA180132V0093"

        print(random_id, first_nr, last_nr)

        _point, _footprint = derive_image_position(random_id, mode=_mode, min_nr_of_images=2,
                                                   polygon_mode='outline',
                                                   polynomial_order=1, verbose=True)
