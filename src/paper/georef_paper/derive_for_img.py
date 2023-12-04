import copy
import math
import numpy as np

from shapely.affinity import translate
from shapely.geometry import Point, LineString, Polygon
from shapely.wkt import loads
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import base.connect_to_db as ctd
import base.print_v as p

import display.display_shapes as ds

#regressor = "TheilSen"
regressor = "RANSAC"

debug_display = False
debug_estimate_all = True
debug_show_poly = False


def derive_for_img(image_id, mode="satellite", min_nr_of_images=3,
               max_range=4, polynomial_order=1,
               outlier_mode=False,
               return_line=False,
               catch=True, verbose=False, pbar=None):
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
                    (data['footprint_type'] == 'satellite')
    elif mode == "images":
        condition = (data['footprint_exact'].notnull()) & \
                    (data['footprint_type'] == 'image')
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
    id_list_complete = id_arr.tolist()
    id_list = [item[-4:] for item in id_list_complete]
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

    def _fit_line(_x, _y):

        # fit model
        try:
            if regressor == "TheilSen":
                _model = make_pipeline(PolynomialFeatures(polynomial_order), TheilSenRegressor(fit_intercept=False))
            elif regressor == "RANSAC":
                from sklearn.linear_model import RANSACRegressor, LinearRegression
                ransac = RANSACRegressor(estimator=LinearRegression(),
                                         max_trials=100,  # Maximum number of iterations
                                         residual_threshold=1,
                                         min_samples=0.5,
                                         random_state=1)
                _model = make_pipeline(PolynomialFeatures(polynomial_order), ransac)
            _model.fit(_x, _y.ravel())
        except (Exception,) as e:
            if return_line:
                return None, None, None
            else:
                return None, None

        # code to create the regression line (for displaying)
        _regression_line_x = np.linspace(min(x), max(x), 100)
        _regression_line_x_standardized = x_scaler.transform(_regression_line_x[..., None])
        _regression_line_y_standardized = _model.predict(_regression_line_x_standardized)
        _regression_line_y = y_scaler.inverse_transform(_regression_line_y_standardized.reshape(-1, 1)).ravel()
        _regression_line = LineString(zip(_regression_line_x, _regression_line_y))
        _regression_line = _regression_line.simplify(100)
        return _model, _regression_line

    model, regression_line = _fit_line(x_train, y_train)

    def _get_residual_outliers(_x, _y, _model):

        # Standardize the x values for prediction
        _x_standardized = x_scaler.transform(x[..., None])

        # Predict using the standardized x values
        _predicted_y_standardized = _model.predict(_x_standardized)

        # Inverse transform the predicted values to get them in original scale
        _predicted_y = y_scaler.inverse_transform(_predicted_y_standardized.reshape(-1, 1)).ravel()

        # Compute residuals of the exact polygons
        _residuals = _y - _predicted_y

        # Compute outlier threshold based on IQR
        _q1 = np.percentile(_residuals, 25)
        _q3 = np.percentile(_residuals, 75)
        _iqr = _q3 - _q1
        _lower_bound = _q1 - 1.5 * _iqr
        _upper_bound = _q3 + 1.5 * _iqr

        # Identify outliers
        _outliers = np.where((_residuals < _lower_bound) | (_residuals > _upper_bound))[0]

        return _outliers.tolist()

    def _get_distance_outliers(_ids, _points):

        # copy to not change
        _ids = copy.deepcopy(_ids)
        _points = copy.deepcopy(_points)

        # required for later
        _orig_ids = copy.deepcopy(_ids)

        # all outliers we've found
        _all_outlier_ids = []

        while True:
            # Compute pairwise distances
            _distances = []
            for i in range(len(_ids) - 1):
                _id1 = _ids[i]
                _id2 = _ids[i + 1]

                _point1 = _points[i]
                _point2 = _points[i + 1]

                # Compute the difference in x-coordinates
                _distance = _point1.distance(_point2)

                # Adjust for gap in IDs
                _gap = abs(_id1 - _id2)

                if _gap > 0:
                    _distance /= _gap

                _distances.append(_distance)

            # Compute the median and 10% of the median
            median_value = np.median(_distances)
            threshold = 0.25 * median_value

            # List to store the involved IDs with too high distances
            involved_ids = []

            # Iterate through the list of distances
            for i, dist in enumerate(_distances):
                if abs(dist - median_value) > threshold:
                    # Add the IDs corresponding to the current and next index
                    if i < len(_ids) - 1:
                        involved_ids.append(_ids[i])
                    if i + 1 < len(_ids):
                        involved_ids.append(_ids[i + 1])

            from collections import Counter
            _id_counts = Counter(involved_ids)
            _outlier_ids = [id_ for id_, count in _id_counts.items() if count == 2]

            # no outliers -> we can stop the loop
            if len(_outlier_ids) == 0:
                break

            # Remove the first outlier ID and its corresponding points
            index = _ids.index(_outlier_ids[0])
            del _ids[index]
            del _points[index]

            _all_outlier_ids = _all_outlier_ids + _outlier_ids

        _outliers = [_orig_ids.index(value) for value in _all_outlier_ids]
        return _outliers

    res_outlier = _get_residual_outliers(x, y, model)
    res_outlier_ids_int = [item for i, item in enumerate(id_list_int) if i in res_outlier]
    res_outlier_points = [item for i, item in enumerate(point_list) if i in res_outlier]
    res_outlier_footprints = [item for i, item in enumerate(footprint_list) if i in res_outlier]

    # remove outliers from lists based on residuals
    cleaned_ids = [item for i, item in enumerate(id_list) if i not in res_outlier]
    cleaned_ids_int = [item for i, item in enumerate(id_list_int) if i not in res_outlier]
    cleaned_points = [item for i, item in enumerate(point_list) if i not in res_outlier]
    cleaned_footprints = [item for i, item in enumerate(footprint_list) if i not in res_outlier]

    # remove outliers based on distance
    dis_outlier = _get_distance_outliers(cleaned_ids_int, cleaned_points)

    dis_outlier_ids_int = [item for i, item in enumerate(cleaned_ids_int) if i in dis_outlier]
    dis_outlier_points = [item for i, item in enumerate(cleaned_points) if i in dis_outlier]
    dis_outlier_footprints = [item for i, item in enumerate(cleaned_footprints) if i in dis_outlier]

    cleaned_ids = [item for i, item in enumerate(cleaned_ids) if i not in dis_outlier]
    cleaned_ids_int = [item for i, item in enumerate(cleaned_ids_int) if i not in dis_outlier]
    cleaned_points = [item for i, item in enumerate(cleaned_points) if i not in dis_outlier]
    cleaned_footprints = [item for i, item in enumerate(cleaned_footprints) if i not in dis_outlier]

    outliers = res_outlier + dis_outlier
    outlier_ids = [id_list_complete[i] for i in outliers]
    outlier_ids_int = res_outlier_ids_int + dis_outlier_ids_int
    outlier_points = res_outlier_points + dis_outlier_points
    outlier_footprints = res_outlier_footprints + dis_outlier_footprints

    if outlier_mode:
        if image_id in outlier_ids:
            return True
        else:
            return False

    if len(cleaned_ids) < 2:
        p.print_v("There are not enough geo-referenced images to derive image positions "
                  "(after outlier filtering) for flight {flight_path}", verbose=verbose, pbar=pbar)
        if return_line:
            return "not_enough_images", None, None
        else:
            return "not_enough_images", None

    # function to find the angle of the regression line
    def find_angle_with_vertical(_linestring):
        # Extract the coordinates of the two endpoints of the LineString
        _x1, _y1 = _linestring.coords[0]
        _x2, _y2 = _linestring.coords[-1]

        def __find_angle(__x1, __y1, __x2, __y2):
            # Handle the vertical line case
            if __x2 == __x1:
                return 90  # or -90, depending on convention

            # Calculate the slope
            __m = (__y2 - __y1) / (__x2 - __x1)

            # Calculate the angle in radians
            __theta_radians = math.atan(__m)

            # Convert to degrees
            __theta_degrees = math.degrees(__theta_radians)

            # Adjust for quadrant if necessary
            if __x2 < __x1:
                __theta_degrees += 180

            # Keep the angle in the range [0, 360]
            __theta_degrees = __theta_degrees % 360

            return __theta_degrees

        # Calculate the angle with the _x-axis
        _angle_x_axis = __find_angle(_x1, _y1, _x2, _y2)

        # Calculate the angle with the y-axis (vertical line)
        if _angle_x_axis <= 90:
            _angle_vertical = 90 - _angle_x_axis
        else:
            _angle_vertical = _angle_x_axis - 90

        return _angle_vertical

    # we need the angle of regression line to later rotate the footprints accordingly
    angle = find_angle_with_vertical(regression_line)

    # calculate the avg distance between one polygon
    def _average_point_distance_list(_ids, _points, _id_int):

        # Compute pairwise distances
        _distances = []
        for i in range(len(_ids) - 1):
            _id1 = _ids[i]
            _id2 = _ids[i + 1]

            _point1 = _points[i]
            _point2 = _points[i + 1]

            # Compute the difference in x-coordinates
            _distance = abs(_point1.x - _point2.x)

            # Adjust for gap in IDs
            _gap = abs(_id1 - _id2)

            if _gap > 0:
                _distance /= _gap

            _distances.append(_distance)

        # Compute average
        return sum(_distances) / len(_distances) if _distances else 0

    # create an average polygon
    def _create_average_polygon(_poly1_lst, _poly2_lst, _weight1, _weight2, _angle):

        # function to translate polygon to origin
        def __translate_to_origin(__polygon):
            __centroid = __polygon.centroid
            __translated_polygon = translate(__polygon, xoff=-__centroid.x, yoff=-__centroid.y)
            return __translated_polygon

        # translate all polygons in both lists to origin
        _poly1_lst_translated = [__translate_to_origin(poly) for poly in _poly1_lst]
        _poly2_lst_translated = [__translate_to_origin(poly) for poly in _poly2_lst]

        # get the avg:
        avg_coords_x = [0, 0, 0, 0, 0]
        avg_coords_y = [0, 0, 0, 0, 0]
        counter = 0
        for _lst in [_poly1_lst_translated, _poly2_lst_translated]:
            for _poly in _lst:
                counter = counter + 1
                start_simplify = 100
                while len(_poly.exterior.coords) > 5:
                    _poly = _poly.simplify(start_simplify, preserve_topology=False)
                    start_simplify += 100

                for j, _coord in enumerate(_poly.exterior.coords):
                    avg_coords_x[j] = avg_coords_x[j] + _coord[0]
                    avg_coords_y[j] = avg_coords_y[j] + _coord[1]

        avg_coords_x = [_x / counter for _x in avg_coords_x]
        avg_coords_y = [_y / counter for _y in avg_coords_y]

        avg_coords = list(zip(avg_coords_x, avg_coords_y))

        # Create a shapely polygon
        avg_polygon = Polygon(avg_coords)

        return avg_polygon

    if debug_estimate_all:
        unref_data = data[condition == False]
        unref_ids = unref_data['image_id'].tolist()
        unref_ids = [item[-4:] for item in unref_ids]
        ids_to_estimate = [int(s) for s in unref_ids]
    else:
        ids_to_estimate = [id_int]

    estimated_ids = []
    estimated_points = []
    estimated_footprints = []

    # the closest position of the real points (required for sorting)
    cleaned_points_closest = []

    # Extract x-coordinates of cleaned_points
    x_coords = [point.x for point in cleaned_points]

    # Get the min and max x-coordinates
    x_min, x_max = min(x_coords) - 1e6, max(x_coords) + 1e6

    # Calculate the slope and y-intercept of the regression line
    start, end = regression_line.coords
    m = (end[1] - start[1]) / (end[0] - start[0])
    c = start[1] - m * start[0]

    # Compute the corresponding y values for the extended x values
    y_min = m * x_min + c
    y_max = m * x_max + c

    # Create the extended regression line
    extended_regression_line = LineString([(x_min, y_min), (x_max, y_max)])
    from shapely.ops import nearest_points
    for point in cleaned_points:
        closest_point_on_line = nearest_points(extended_regression_line, point)[0]
        cleaned_points_closest.append(closest_point_on_line)

    # check if already have id_int
    if id_int in cleaned_ids_int:
        idx = cleaned_ids_int.index(id_int)
        estimated_ids.append(id_int)
        estimated_poly = cleaned_footprints[idx]
        estimated_point = cleaned_points[idx]

    for id_int in ids_to_estimate:

        # get closest ids and polygons
        smaller_ids = sorted([num for num in cleaned_ids_int if num < id_int], key=lambda n: abs(id_int - n))
        larger_ids = sorted([num for num in cleaned_ids_int if num > id_int], key=lambda n: abs(id_int - n))

        smaller_ids = smaller_ids[:max_range]
        larger_ids = larger_ids[:max_range]

        smaller_footprints = [cleaned_footprints[cleaned_ids_int.index(i)] for i in smaller_ids]
        larger_footprints = [cleaned_footprints[cleaned_ids_int.index(i)] for i in larger_ids]
        smaller_points = [cleaned_points[cleaned_ids_int.index(i)] for i in smaller_ids]
        larger_points = [cleaned_points[cleaned_ids_int.index(i)] for i in larger_ids]
        smaller_points_closest = [cleaned_points_closest[cleaned_ids_int.index(i)] for i in smaller_ids]
        larger_points_closest = [cleaned_points_closest[cleaned_ids_int.index(i)] for i in larger_ids]

        # sort smaller by x of regression line
        if smaller_ids:
            sorted_data = sorted(zip(smaller_ids, smaller_footprints, smaller_points, smaller_points_closest), key=lambda item: item[3].x)
            smaller_ids, smaller_footprints, smaller_points, smaller_points_closest = zip(*sorted_data)
            smaller_ids = list(smaller_ids)
            smaller_footprints = list(smaller_footprints)
            smaller_points = list(smaller_points)
            smaller_points_closest = list(smaller_points_closest)

        # sort larger by x of regression line
        if larger_ids:
            sorted_data = sorted(zip(larger_ids, larger_footprints, larger_points, larger_points_closest), key=lambda item: item[3].x)
            larger_ids, larger_footprints, larger_points, larger_points_closest = zip(*sorted_data)
            larger_ids = list(larger_ids)
            larger_footprints = list(larger_footprints)
            larger_points = list(larger_points)
            larger_points_closest = list(larger_points_closest)

        if len(smaller_ids) > 0 and len(larger_ids) > 0:

            # we just have one direction
            if len(smaller_ids) > 1:
                if smaller_ids[0] < smaller_ids[-1]:
                    direction_ids = "left_smaller_right_bigger"
                elif smaller_ids[0] > smaller_ids[-1]:
                    direction_ids = "left_bigger_right_smaller"
                else:
                    raise ValueError(f"direction_ids wrong for {flight_path}")
            elif len(larger_ids) > 1:
                if larger_ids[0] < larger_ids[-1]:
                    direction_ids = "left_smaller_right_bigger"
                elif larger_ids[0] > larger_ids[-1]:
                    direction_ids = "left_bigger_right_smaller"
                else:
                    raise ValueError(f"direction_ids wrong for {flight_path}")
            else:
                direction_ids = "left_smaller_right_bigger"

            # get distance between smallest and largest in ids and px
            if direction_ids == "left_smaller_right_bigger":
                id_distance = id_int - smaller_ids[-1]
                lst_id_distance = larger_ids[0] - smaller_ids[-1]
                distance = larger_points_closest[0].x - smaller_points_closest[-1].x
                avg_distance = distance / lst_id_distance

                estimated_x = smaller_points_closest[-1].x + id_distance * avg_distance

            elif direction_ids == "left_bigger_right_smaller":
                id_distance = id_int - smaller_ids[0]
                lst_id_distance = larger_ids[-1] - smaller_ids[0]
                distance = larger_points_closest[-1].x - smaller_points_closest[0].x
                avg_distance = np.abs(distance / lst_id_distance)

                estimated_x = smaller_points_closest[0].x - id_distance * avg_distance

            smaller_weight = 1
            larger_weight = 1

        elif len(smaller_ids) > 0:
            if smaller_ids[0] < smaller_ids[-1]:
                direction_smaller_ids = "left_smaller_right_bigger"
            else:
                direction_smaller_ids = "left_bigger_right_smaller"

            # get avg px distance in lst
            avg_small_distance = _average_point_distance_list(smaller_ids, smaller_points_closest, id_int)

            # get distance of target to closest in ids and px
            if direction_smaller_ids == "left_smaller_right_bigger":
                small_id_distance = id_int - smaller_ids[-1]
                small_distance = avg_small_distance * small_id_distance
                estimated_x = smaller_points_closest[-1].x + small_distance

            elif direction_smaller_ids == "left_bigger_right_smaller":
                small_id_distance = id_int - smaller_ids[0]
                small_distance = avg_small_distance * small_id_distance
                estimated_x = smaller_points_closest[0].x - small_distance

            smaller_weight = 1
            larger_weight = 0

        elif len(larger_ids) > 0:
            if larger_ids[0] < larger_ids[-1]:
                direction_larger_ids = "left_smaller_right_bigger"
            else:
                direction_larger_ids = "left_bigger_right_smaller"

            # get avg px distance in lst
            avg_large_distance = _average_point_distance_list(larger_ids, larger_points_closest, id_int)

            # get distance of target to largest in ids
            if direction_larger_ids == "left_smaller_right_bigger":
                large_id_distance = larger_ids[0] - id_int
                large_distance = avg_large_distance * large_id_distance
                estimated_x = larger_points_closest[0].x - large_distance

            elif direction_larger_ids == "left_bigger_right_smaller":
                large_id_distance = larger_ids[-1] - id_int
                large_distance = avg_large_distance * large_id_distance
                estimated_x = larger_points_closest[-1].x + large_distance

            larger_weight = 1
            smaller_weight = 0
        else:
            raise ValueError("No entry in smaller or bigger")

        # get estimated y
        # get estimated y
        x_standardized = x_scaler.transform(np.array(estimated_x).reshape(-1, 1))  # noqa

        # calc y
        y_standardized = np.array(model.predict(x_standardized))
        estimated_y = y_scaler.inverse_transform(y_standardized.reshape(-1, 1)).ravel()[0]

        # create estimated point
        estimated_point = Point(estimated_x, estimated_y)
        estimated_points.append(estimated_point)

        # create estimated footprint
        estimated_poly = _create_average_polygon(smaller_footprints, larger_footprints,
                                                 smaller_weight, larger_weight, angle)  # noqa

        # move polygon to the right position
        dx = estimated_x - estimated_poly.centroid.x
        dy = estimated_y - estimated_poly.centroid.y

        estimated_poly = translate(estimated_poly, xoff=dx, yoff=dy)
        estimated_footprints.append(estimated_poly)
        estimated_ids.append(id_int)

    filtered_data = [(id_val, footprint, point) for id_val, footprint, point in zip(cleaned_ids, cleaned_footprints, cleaned_points)
                     if first_nr <= int(id_val[-4:]) <= last_nr]
    cleaned_ids, cleaned_footprints, cleaned_points = zip(*filtered_data)
    cleaned_ids = list(cleaned_ids)
    cleaned_footprints = list(cleaned_footprints)
    cleaned_points = list(cleaned_points)

    filtered_data = [(id_val, footprint, point) for id_val, footprint, point in zip(estimated_ids, estimated_footprints, estimated_points)
                     if first_nr <= id_val <= last_nr]
    estimated_ids, estimated_footprints, estimated_points = zip(*filtered_data)
    estimated_ids = list(estimated_ids)
    estimated_footprints = list(estimated_footprints)
    estimated_points = list(estimated_points)

    print(cleaned_ids)
    print(estimated_ids)


    # Extract x-coordinates of cleaned_points
    x_max = estimated_points[0].x
    x_min = cleaned_points[-1].x

    # Calculate the slope and y-intercept of the regression line
    start, end = regression_line.coords
    m = (end[1] - start[1]) / (end[0] - start[0])
    c = start[1] - m * start[0]

    # Compute the corresponding y values for the extended x values
    y_min = m * x_min + c
    y_max = m * x_max + c

    # Create the extended regression line
    regression_line = LineString([(x_min, y_min), (x_max, y_max)])

    if debug_display:
        if debug_show_poly is False:
            cleaned_footprints = []
            outlier_footprints = []
            estimated_footprints = []

        if debug_show_estimated is False:
            estimated_ids = []
            estimated_footprints = []
            estimated_points = []

            #shorten line string
            from shapely.geometry import MultiPoint
            multipoint = MultiPoint(cleaned_points)
            envelope = multipoint.envelope
            regression_line = regression_line.intersection(envelope)

        extend_points = []

        ds.display_shapes([regression_line,
                           cleaned_footprints, cleaned_points,
                           outlier_footprints, outlier_points,
                           estimated_footprints, estimated_points,
                           extend_points],
                          subtitles=[None,
                                     None, cleaned_ids_int,
                                     None, outlier_ids_int,
                                     None, estimated_ids, None],
                          colors=["black", "green", "green", "red", "red", "lightgray", "lightgray", "white"],
                          alphas=[1,1,1,1,1,0.5,1,1],
                          title=flight_path, text_size=13)

    if return_line:
        return estimated_point, estimated_poly, regression_line
    else:
        return estimated_point, estimated_poly


if __name__ == "__main__":

    debug_display = True
    debug_show_poly = True
    debug_show_estimated = True

    flight_paths = [
        1942,2139,1206,1358,1738,1353,1819,1549,2147,1994,2133,5122,1802,1876,
        1721,1748,1806,5125,2140,1963,1725,2043,5126,2134,1803,1744,1724,1747,
        1722,1745,2042,1746,1719,5120,1741,1718,1723,1965,1804,1811,1720,1893,
        5124,5127,1993,2135,1898,1982,2165,1743,1969,2160,1817,1742,1809,1968,
        2163,2167,1807,1966,2161,1814,2041,1829,1967,1805,1834,2141,1962,1830,
        2123,1810,2164,1827,2040,2124,1644,2166,1818,2039,2142,2074,2121,1800,
        1815,1831,1812,1832,1801,1845,1826,1848,1843,2073,5123,2075,1847,1835,
        1844,1849,1823,1846,1824,1822,1825,1813,1833,1816,1821
    ]

    flight_paths = [1812]

    import random
    random.shuffle(flight_paths)

    # flight_paths = [2160, 2140] # , 2160, 1822]

    for _flight_path in flight_paths:
        _sql_string = f"SELECT image_id FROM IMAGES WHERE SUBSTRING(image_id, 3, 4) ='{_flight_path}' AND " \
                      f"image_id LIKE '%V%' ORDER BY image_id"
        _data = ctd.get_data_from_db(_sql_string, catch=False)

        random_id = _data.sample(1)['image_id'].iloc[0]
        first_nr = int(_data['image_id'].iloc[0][-4:])
        last_nr = int(_data['image_id'].iloc[-1][-4:])

        random_id = "CA181232V0060"
        first_nr = 50
        last_nr = 70

        test = derive_for_img(random_id, min_nr_of_images=3,
                   polynomial_order=1, verbose=False)
