import math

import numpy as np

from shapely.affinity import rotate, translate
from shapely.geometry import Point, LineString, Polygon
from shapely.wkt import loads
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

import base.connect_to_db as ctd
import base.print_v as p

import display.display_shapes as ds

_verbose = True

_mode = "satellite"  # mode can be 'satellite', 'images' or 'both'
_polygon_mode = "outline"  # polygon mode can be 'outline' or 'exact'

debug_display_flight_path = True

debug_show_outliers = False
debug_show_footprints = False

debug_delta = 'all'  # can be all or a number; plot only so many images around it


def derive_image_position(image_id,
                          mode="satellite", polygon_mode="outline",
                          min_nr_of_images=4,
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

    # get all images with this flight path
    sql_string = "SELECT image_id, SUBSTRING(image_id, 3, 4) AS flight_path, " \
                 "ST_AsText(footprint_exact) AS footprint_exact, " \
                 "ST_AsText(footprint_approx) as footprint_approx, " \
                 "ST_AsText(position_exact) AS position_exact, " \
                 "ST_AsText(position_approx) AS position_approx, " \
                 "position_error_vector, footprint_Type FROM images_extracted " \
                 f"WHERE SUBSTRING(image_id, 3, 4) ='{flight_path}' AND " \
                 f"image_id LIKE '%V%'"
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
    id_list = id_arr.tolist()
    id_list = [item[-4:] for item in id_list]
    id_list_int = [int(s) for s in id_list]

    # get all exact positions as a list and convert WKT strings to Shapely Point objects
    wkt_point_list = georef_data['position_exact'].tolist()
    wkt_footprint_list = georef_data['footprint_exact'].tolist()
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
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        model = make_pipeline(PolynomialFeatures(polynomial_order), HuberRegressor())
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

    # remove outliers from list
    x_cleaned = [item for i, item in enumerate(x) if i not in outliers]
    y_cleaned = [item for i, item in enumerate(y) if i not in outliers]

    # remove outliers from list: part 2
    cleaned_ids = [item for i, item in enumerate(id_list) if i not in outliers]
    cleaned_ids_int = [item for i, item in enumerate(id_list_int) if i not in outliers]
    cleaned_points = [item for i, item in enumerate(point_list) if i not in outliers]
    cleaned_footprints = [item for i, item in enumerate(footprint_list) if i not in outliers]

    # get outliers
    outlier_ids = [item for i, item in enumerate(id_list) if i in outliers]
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

    # we need the angle to later rotate the footprints accordingly
    angle = find_angle_with_vertical(regression_line)

    # Find gaps in the list
    gaps = [(cleaned_ids_int[i], cleaned_ids_int[i + 1]) for i in range(len(cleaned_ids_int) - 1) if
            cleaned_ids_int[i + 1] - cleaned_ids_int[i] > 1]

    # here we will save the estimated values
    estimated_ids_int = []  # the ids of every estimated image
    estimated_ids = []  # the ids of every estimated image
    estimated_points = []  # the position as shapely point
    estimated_footprints = []  # the approx_footprint of this position

    # here we save the distances and directions per gap
    gap_distances = []
    gap_directions = []

    # iterate all gaps
    for gap in gaps:

        # identify the start and end of a gap
        start_x, start_y = x_cleaned[cleaned_ids_int.index(gap[0])], y_cleaned[cleaned_ids_int.index(gap[0])]
        end_x, end_y = x_cleaned[cleaned_ids_int.index(gap[1])], y_cleaned[cleaned_ids_int.index(gap[1])]
        num_points_between = gap[1] - gap[0] - 1
        total_distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        avg_distance_local = total_distance / (num_points_between + 1)

        # save distance
        gap_distances.append(avg_distance_local)

        # Direction vector between the two points
        direction_x = (end_x - start_x) / total_distance

        # save direction
        gap_directions.append(direction_x)

        # Using the direction vector and local average distance to compute steps the estimated position
        x_step = direction_x * avg_distance_local
        for j in range(num_points_between):
            # calc _x
            x_pos = start_x + x_step * (j + 1)
            x_standardized = x_scaler.transform(np.array(x_pos).reshape(-1, 1))

            # calc y
            y_standardized = np.array(model.predict(x_standardized))
            y_pos = y_scaler.inverse_transform(y_standardized.reshape(-1, 1)).ravel()[0]

            # add shapely point to list
            estimated_points.append(Point(x_pos, y_pos))

        # for every gap add the id
        for i in range(gap[0] + 1, gap[1]):
            estimated_ids_int.append(i)
            estimated_ids.append(str(i).zfill(4))

    if len(gap_directions) == 0:
        if return_line:
            return "not_enough_images", None, None
        else:
            return "not_enough_images", None

    # special case if the image position we want is smaller
    if int(image_id[-4:]) < cleaned_ids_int[0]:

        # how many new points do we need?
        num_new_points = cleaned_ids_int[0] - int(image_id[-4:])

        # Using the direction vector and local average distance to compute steps the estimated position
        x_step = gap_directions[0] * gap_distances[0]

        # get _x of our starting point
        start_x = x_cleaned[0]

        # calculate the new points
        for j in range(num_new_points):
            # calc _x
            x_pos = start_x - x_step * (j + 1)

            x_standardized = x_scaler.transform(np.array(x_pos).reshape(-1, 1))

            # calc y
            y_standardized = np.array(model.predict(x_standardized))
            y_pos = y_scaler.inverse_transform(y_standardized.reshape(-1, 1)).ravel()[0]

            # add shapely point and ids to list
            estimated_points.append(Point(x_pos, y_pos))
            estimated_ids_int.append(cleaned_ids_int[0] - j - 1)
            estimated_ids.append(str(cleaned_ids_int[0] - j - 1).zfill(4))

    # other special case if the image position we want is higher
    elif int(image_id[-4:]) > cleaned_ids_int[-1]:

        # how many new points do we need?
        num_new_points = int(image_id[-4:]) - cleaned_ids_int[-1]

        # Using the direction vector and local average distance to compute steps the estimated position
        x_step = gap_directions[-1] * gap_distances[-1]

        # get _x of our starting point
        start_x = x_cleaned[-1]

        # calculate the new points
        for j in range(num_new_points):
            # calc _x
            x_pos = start_x + x_step * (j + 1)
            x_standardized = x_scaler.transform(np.array(x_pos).reshape(-1, 1))

            # calc y
            y_standardized = np.array(model.predict(x_standardized))
            y_pos = y_scaler.inverse_transform(y_standardized.reshape(-1, 1)).ravel()[0]

            # add shapely point and ids to list
            estimated_points.append(Point(x_pos, y_pos))
            estimated_ids_int.append(cleaned_ids_int[-1] + j + 1)
            estimated_ids.append(str(cleaned_ids_int[-1] + j + 1).zfill(4))

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

    polygon_size = find_polygon_size(cleaned_footprints)

    # function to create an average polygon
    def average_polygon(_poly1, _poly2, distances=(1, 1)):

        # convert distances to weights
        weights = (distances[1] / (distances[0] + distance_1),
                   distances[0] / (distances[0] + distance_1))

        # Ensure both polygons have the same number of vertices
        if len(_poly1.exterior.coords) != len(_poly2.exterior.coords):
            # raise ValueError("The two polygons must have the same number of vertices.")
            return None

        # Ensure weights sum up to 1
        if not (0.99 <= sum(weights) <= 1.01):
            raise ValueError("Weights must sum up to 1.")

        def sort_vertices(polygon):

            def compute_angle(point, _centroid):
                # Returns the angle (in radians) of a point with respect to a centroid
                return math.atan2(point[1] - _centroid[1], point[0] - _centroid[0])

            # Sorts the vertices of a polygon based on their angle with respect to the centroid
            centroid = polygon.centroid.coords[0]
            sorted_coords = sorted(list(polygon.exterior.coords)[:-1], key=lambda _p: compute_angle(_p, centroid))
            sorted_coords.append(sorted_coords[0])  # close the polygon
            return sorted_coords

        sorted_coords1 = sort_vertices(_poly1)
        sorted_coords2 = sort_vertices(_poly2)

        # Calculate the weighted average coordinates for the new polygon
        averaged_coords = []
        for (x1, y1), (x2, y2) in zip(sorted_coords1, sorted_coords2):
            avg_x = (x1 * weights[0] + x2 * weights[1])
            avg_y = (y1 * weights[0] + y2 * weights[1])
            averaged_coords.append((avg_x, avg_y))

        # Construct the new polygon
        new_poly = Polygon(averaged_coords)

        return new_poly

    # get the closest polygons
    def get_closest_polygons(target, _ids, polygon_lst):

        for i in range(len(_ids)):
            if _ids[i] > target:
                bordering_idxs = [i - 1, i] if i - 1 >= 0 else [i]
                break
        else:
            bordering_idxs = [len(_ids) - 1]

        if len(bordering_idxs) == 1:
            bordering_idxs = bordering_idxs * 2

        # get the polygons & ids
        _bordering_polygons = []
        _bordering_ids = []
        for _idx in bordering_idxs:
            _bordering_polygons.append(polygon_lst[_idx])
            _bordering_ids.append(_ids[_idx])

        return _bordering_polygons, _bordering_ids

    # create a approx_footprint for each estimate point
    for idx, point in enumerate(estimated_points):

        # Assuming point contains the derived position
        center_x, center_y = point.x, point.y

        if polygon_mode == "outline":
            # Construct the rotated square using the average size and rotation
            half_side_length = polygon_size / (2 * np.sqrt(2))
            half_diagonal = half_side_length * np.sqrt(2) * 2

            dx = half_diagonal * np.cos(np.radians(angle))
            dy = half_diagonal * np.sin(np.radians(angle))
            poly = Polygon([
                (center_x + dx, center_y + dy),
                (center_x - dy, center_y + dx),
                (center_x - dx, center_y - dy),
                (center_x + dy, center_y - dx)
            ])

        elif polygon_mode == "exact":

            # get the closest two polygons and calc average
            closest_polygons, closest_ids_int = get_closest_polygons(estimated_ids_int[idx],
                                                                     cleaned_ids_int,
                                                                     cleaned_footprints)

            distance_0 = np.abs(closest_ids_int[0] - estimated_ids_int[idx])
            distance_1 = np.abs(closest_ids_int[1] - estimated_ids_int[idx])

            def simplify_to_square(polygon):

                # Compute the convex hull of the polygon
                hull = polygon.convex_hull

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
                    rotated_poly = rotate(polygon, _angle, origin='centroid')

                    # Compute the bounding box for the rotated polygon
                    _min_x, _min_y, _max_x, _max_y = rotated_poly.bounds
                    side_length = max(_max_x - _min_x, _max_y - _min_y)
                    square = Polygon([(_min_x, _min_y), (_min_x + side_length, _min_y),
                                      (_min_x + side_length, _min_y + side_length), (min_x, _min_y + side_length)])

                    # Check if this square is the smallest encountered so far
                    if square.area < min_area:
                        min_area = square.area
                        best_square = square
                        best_angle = angle

                # Rotate the best square back to align with the hull edge's original orientation
                best_square = rotate(best_square, -best_angle, origin='centroid')

                return best_square

            # simplify polygons
            poly1 = simplify_to_square(closest_polygons[0])
            poly2 = simplify_to_square(closest_polygons[1])

            avg_poly = average_polygon(poly1, poly2,
                                       distances=(distance_0, distance_1))

            if avg_poly is None:
                continue

            # move polygon to the right position
            dx = point.x - avg_poly.centroid.x
            dy = point.y - avg_poly.centroid.y

            poly = translate(avg_poly, xoff=dx, yoff=dy)

        else:
            raise ValueError(f"polygon_mode '{polygon_mode}' is not valid")

        estimated_footprints.append(poly)

    # display the flight path
    if debug_show_footprints:

        # check if we want to show outliers
        if debug_show_outliers is False:
            outlier_footprints = []
            outlier_points = []

        # we want to show all
        if debug_delta == "all":

            # we can just copy the lists
            display_regression_line = regression_line
            display_cleaned_ids = cleaned_ids
            display_cleaned_footprints = cleaned_footprints
            display_cleaned_points = cleaned_points
            display_estimated_ids = estimated_ids
            display_estimated_footprints = estimated_footprints
            display_estimated_points = estimated_points
            display_outlier_ids = outlier_ids
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
                    display_cleaned_ids.append(cleaned_ids[idx])
                    display_cleaned_footprints.append(cleaned_footprints[idx])
                    display_cleaned_points.append(cleaned_points[idx])

            # filter estimated
            for idx, elem in enumerate(estimated_ids_int):
                if elem in display_nrs:
                    display_estimated_ids.append(estimated_ids[idx])
                    display_estimated_footprints.append(estimated_footprints[idx])
                    display_estimated_points.append(estimated_points[idx])

            # filter outlier
            for idx, elem in enumerate(outlier_ids_int):
                if elem in display_nrs:
                    display_outlier_ids.append(outlier_ids[idx])
                    display_outlier_footprints.append(outlier_footprints[idx])
                    display_outlier_points.append(outlier_points[idx])

            # get min and max _x from the points
            combined_points = display_cleaned_points + display_estimated_points + display_outlier_points
            x_values = [point.x for point in combined_points]
            min_x, max_x = min(x_values), max(x_values)

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
    flight_paths = [
        1962, 1849, 1846, 2163, 1719, 1817, 1827, 1844, 1803, 1644, 1825, 1833, 2121, 1832, 1816, 1834, 1845, 1829,
        2134, 1804, 1963, 1969, 1742, 2140, 1965, 1826, 1801, 2133, 1745, 1814, 2135, 2040, 1876, 2167, 1741, 1822,
        1724, 1744, 1847, 2139, 1942, 1893, 1721, 5127, 5120, 1982, 1720, 2041, 5124, 2042, 1898, 2039, 2043, 5126,
        5125, 1725, 1723, 1994, 1802, 1743, 1810, 1823, 1830, 1815, 1807, 1843]

    debug_show_footprints = True
    debug_show_outliers = False

    import random
    random.shuffle(flight_paths)

    # flight_paths = [1801]
    # flight_paths = [1848]
    flight_paths = [1823]

    for _flight_path in flight_paths:
        _sql_string = f"SELECT image_id FROM IMAGES WHERE SUBSTRING(image_id, 3, 4) ='{_flight_path}' AND " \
                      f"image_id LIKE '%V%'"
        _data = ctd.get_data_from_db(_sql_string, catch=False)

        print(f"There are {_data.shape[0]} images for this flight path")

        random_id = _data.sample(1)['image_id'].iloc[0]
        # random_id = _data['image_id'][0]
        # random_id = "CA180132V0093"

        _point, _footprint = derive_image_position(random_id, mode=_mode, min_nr_of_images=2,
                                                   polygon_mode='exact',
                                                   polynomial_order=2, verbose=True)

        print("")
