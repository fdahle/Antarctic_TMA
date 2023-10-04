import numpy as np

from shapely.geometry import LineString
from shapely.wkt import loads
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

import base.connect_to_db as ctd
import base.print_v as p

import display.display_shapes as ds

debug_display_outliers = False


def get_invalid_position(image_id, mode="satellite", min_nr_of_images=3, threshold_val=50,
                         catch=True, verbose=False, pbar=None):
    """
    get_invalid_position(image_id, mode, min_nr_of_images, catch, verbose, pbar)
    This function uses the image id to check if the position of an image is valid. It converts the id to
    a flight path, creates a regression line for this flight path and looks if the image is roughly on
    this line.
    Args:
        image_id (String): The id of the image for which we want to check the position
        mode (String, 'satellite'):
        min_nr_of_images (int, 3):
        threshold_val (int, 50):
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:

    """

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
    else:
        raise ValueError("Mode '{mode} is not valid'")

    # create a subset with geo-referenced data
    georef_data = data[condition]

    # check how many images we have in our flight path
    nr_georef_images = georef_data.shape[0]

    # do we have enough images
    if nr_georef_images < min_nr_of_images:
        p.print_v("There are not enough geo-referenced images to derive image positions "
                  f"for flight {flight_path}", verbose=verbose, pbar=pbar)

        # for flight paths with not enough images we assume the image is correct
        return True

    # order data-frame
    georef_data = georef_data.sort_values(by="image_id")

    # get all ids from flight-path as a list and convert list to integers
    id_list = georef_data["image_id"].tolist()
    id_list = [item[-4:] for item in id_list]

    # get all exact positions as a list and convert WKT strings to Shapely Point objects
    wkt_point_list = georef_data['position_exact'].tolist()
    point_list = [loads(wkt) for wkt in wkt_point_list]

    # Extract x and y coordinates using list comprehension
    x = np.array([point.x for point in point_list])
    y = np.array([point.y for point in point_list])

    # standardize
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(x[..., None])
    y_train = y_scaler.fit_transform(y[..., None])

    # fit model
    try:
        model = HuberRegressor(epsilon=1)
        model.fit(x_train, y_train.ravel())
    except (Exception,) as e:
        if catch:
            return False
        else:
            raise e

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

    # ensure that a small deviation is still allowed
    lower_bound = min(lower_bound, -threshold_val)
    upper_bound = max(upper_bound, threshold_val)

    # convert to line
    regression_line = LineString(zip(regression_line_x, regression_line_y))

    # Identify outliers
    outliers = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]

    # Sort the indices in reverse order to avoid index shifting issues
    sorted_outliers = sorted(outliers, reverse=True)

    outlier_list = []
    outlier_id_list = []

    # Pop elements from the list based on the sorted indices
    for index in sorted_outliers:
        if 0 <= index < len(id_list):
            outlier_list.append(point_list.pop(index))
            outlier_id_list.append(id_list.pop(index))

    if debug_display_outliers:
        ds.display_shapes([point_list, outlier_list, regression_line],
                          subtitles=[id_list, outlier_id_list, ""],
                          colors=["green", "red", "lightgray"],
                          title=f"Outliers for {flight_path}")

    # check if outlier in list
    if str(image_id[-4:]) in outlier_id_list:
        outlier = True
    else:
        outlier = False

    # check if valid
    if outlier:
        validity = False
    else:
        validity = True

    return validity
