import base.connect_to_db as ctd
import base.print_v as p

debug_more_details = True

debug_focal_length_accuracy = 3


def quality_check_table_extracted():

    # get data from table (cam_ids per flight)
    sql_string = f"SELECT SUBSTRING(image_id, 3, 7) AS flight_path, " \
                 f"cam_id " \
                 f"FROM images_extracted " \
                 f"GROUP BY flight_path, cam_id"
    data = ctd.get_data_from_db(sql_string, catch=False)

    # check for incorrect cam ids (different ones for the same flight)
    data_no_nan = data.dropna()
    incorrect_values = data_no_nan[data_no_nan.duplicated(subset=['cam_id'])]['cam_id'].unique()
    percentage_incorrect_values = round(incorrect_values.shape[0] / (data.shape[0]) * 100, 2)
    if incorrect_values.shape[0] == 0:
        p.print_v(f"No cam ids are mismatching for flight paths", color="green")
    else:
        p.print_v(f"Cam ids are mismatching for {incorrect_values.shape[0]}/{data.shape[0]} "
                  f"({percentage_incorrect_values}%) flight paths", color="red")
        if debug_more_details:
            p.print_v(incorrect_values.tolist())

    # check for no cam ids
    nan_values = data.loc[data['cam_id'].isnull(), 'flight_path']
    percentage_nan_values = round(nan_values.shape[0] / (data.shape[0]) * 100, 2)
    if nan_values.shape[0] == 0:
        p.print_v(f"No cam ids are missing for flight paths", color="green")
    else:
        p.print_v(f"Cam ids are missing for {nan_values.shape[0]}/{data.shape[0]} "
                  f"({percentage_nan_values}%) flight paths", color="red")
        if debug_more_details:
            p.print_v(nan_values.tolist())

    # get data from table (focal lengths per flight)
    sql_string = f"SELECT SUBSTRING(image_id, 3, 7) AS flight_path, " \
                 f"focal_length " \
                 f"FROM images_extracted " \
                 f"GROUP BY flight_path, focal_length"
    data = ctd.get_data_from_db(sql_string, catch=False)

    data['focal_length'] = data['focal_length'].round(debug_focal_length_accuracy)

    # check for incorrect focal lengths (different ones for the same flight)
    data_no_nan = data.dropna()
    incorrect_values = data_no_nan[data_no_nan.duplicated(subset=['flight_path'])]['flight_path'].unique()
    percentage_incorrect_values = round(incorrect_values.shape[0] / (data.shape[0]) * 100, 2)
    if incorrect_values.shape[0] == 0:
        p.print_v(f"No focal lengths are mismatching for flight paths", color="green")
    else:
        p.print_v(f"Focal lengths are mismatching for {incorrect_values.shape[0]}/{data.shape[0]} "
                  f"({percentage_incorrect_values}%) flight paths", color="red")
        if debug_more_details:
            p.print_v(incorrect_values.tolist())

    # check for no focal lengths
    nan_values = data.loc[data['focal_length'].isna(), 'flight_path']
    percentage_nan_values = round(nan_values.shape[0] / (data.shape[0]) * 100, 2)
    if nan_values.shape[0] == 0:
        p.print_v(f"No focal lengths are missing for flight paths", color="green")
    else:
        p.print_v(f"Focal lengths are missing for {nan_values.shape[0]}/{data.shape[0]} "
                  f"({percentage_nan_values}%) flight paths", color="red")
        if debug_more_details:
            p.print_v(nan_values.tolist())

    # check for focal lengths outside 5%
    avg_focal_length = data['focal_length'].mean(skipna=True)
    outside_values = data[(data['focal_length'] < avg_focal_length * 0.95) |
                          (data['focal_length'] > avg_focal_length * 1.05)]['flight_path']
    percentage_outside_values = round(outside_values.shape[0] / (data.shape[0]) * 100, 2)
    if outside_values.shape[0] == 0:
        p.print_v(f"No focal lengths are wrong for flight paths", color="green")
    else:
        p.print_v(f"Focal lengths are wrong for {outside_values.shape[0]}/{data.shape[0]} "
                  f"({percentage_outside_values}%) flight paths", color="red")
        if debug_more_details:
            p.print_v(outside_values.tolist())

    # check the cam ids
    sql_string = "SELECT cam_id, focal_length FROM images_extracted WHERE cam_id IS NOT NULL " \
                 "GROUP BY cam_id, focal_length"
    data = ctd.get_data_from_db(sql_string)

    # how many cam-ids have no focal length?
    nan_values = data.loc[data['focal_length'].isna(), 'cam_id']
    percentage_nan_values = round(nan_values.shape[0] / (data.shape[0]) * 100, 2)
    if nan_values.shape[0] == 0:
        p.print_v(f"No focal lengths are missing for cam ids", color="green")
    else:
        p.print_v(f"Focal lengths are missing for {nan_values.shape[0]}/{data.shape[0]} "
                  f"({percentage_nan_values}%) cam-ids", color="red")
        if debug_more_details:
            p.print_v(nan_values.tolist())

    # check for incorrect cam-ids (aka having multiple focal length)
    incorrect_values = data[data.duplicated(subset=['cam_id'])]['cam_id'].unique()
    percentage_incorrect_values = round(incorrect_values.shape[0] / (data.shape[0]) * 100, 2)
    if incorrect_values.shape[0] == 0:
        p.print_v(f"No focal lengths are mismatching for cam ids", color="green")
    else:
        p.print_v(f"Focal lengths are mismatching for {incorrect_values.shape[0]}/{data.shape[0]} "
                  f"({percentage_incorrect_values}%) cam-ids", color="red")
        if debug_more_details:
            p.print_v(incorrect_values.tolist())

    # check the focal lengths
    sql_string = "SELECT focal_length, cam_id FROM images_extracted WHERE focal_length IS NOT NULL " \
                 "GROUP BY focal_length, cam_id"
    data = ctd.get_data_from_db(sql_string)

    # how many focal lengths have no cam image_id?
    nan_values = data.loc[data['cam_id'].isna(), 'focal_length']
    percentage_nan_values = round(nan_values.shape[0] / (data.shape[0]) * 100, 2)
    if nan_values.shape[0] == 0:
        p.print_v(f"No cam ids are missing for focal lengths", color="green")
    else:
        p.print_v(f"Cam ids are missing for {nan_values.shape[0]}/{data.shape[0]} "
                  f"({percentage_nan_values}%) focal lengths", color="red")
        if debug_more_details:
            p.print_v(nan_values.tolist())


if __name__ == "__main__":
    quality_check_table_extracted()
