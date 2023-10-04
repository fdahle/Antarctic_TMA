import base.connect_to_db as ctd
import base.print_v as p

debug_more_details = True
debug_max_cutoff = 100


def quality_check_table_fid_points():
    """
    quality_check_table_fid_points():
    This function performs a check on the fid-points stored in the table:
     - are there nan values
     - are there negative values
     - deviate the fid-points too much from each other
    Returns:
        None
    """
    # get data from table
    sql_string = f"SELECT * FROM images_fid_points"
    data = ctd.get_data_from_db(sql_string)

    # iterate all eight fid points directions
    for fid_dir in ["1", "2", "3", "4", "5", "6", "7", "8"]:

        # how many nan values do we have in our data
        number_nan_values_x = data[f"fid_mark_{fid_dir}_x"].isna().sum()
        percentage_nan_x = round(number_nan_values_x / (data.shape[0]) * 100, 2)
        number_nan_values_y = data[f"fid_mark_{fid_dir}_y"].isna().sum()
        missing_ids = data.loc[data[f"fid_mark_{fid_dir}_y"].isna(), 'image_id'].tolist()

        # check if we have same number of entries for x and y
        if number_nan_values_x != number_nan_values_y:
            p.print_v(f"Different number of entries for x/y for fid_mark {fid_dir}", color="red")

        # we have the same number!
        else:

            # check nan values
            if number_nan_values_x == 0:
                p.print_v(f"No fid-marks are missing for fid_mark {fid_dir}", color="green")
            else:
                p.print_v(f"{number_nan_values_x}/{data.shape[0]} ({percentage_nan_x}%) "
                          f"fid-marks are missing for fid_mark {fid_dir}", color="red")

                # show more details
                if debug_more_details:
                    if len(missing_ids) > debug_max_cutoff:
                        p.print_v(missing_ids[0:debug_max_cutoff])
                        p.print_v(f"list was cut off after {debug_max_cutoff} entries")
                    else:
                        p.print_v(missing_ids)

        # check for negative numbers
        negative_rows = data[(data[f"fid_mark_{fid_dir}_x"] < 0) |
                             (data[f"fid_mark_{fid_dir}_y"] < 0)]
        percentage_negative = round(negative_rows.shape[0] / (data.shape[0]) * 100, 2)
        negative_ids = negative_rows['image_id'].tolist()

        # check if we are good or not
        if negative_rows.shape[0] == 0:
            p.print_v(f"No negative values for fid_mark {fid_dir}", color="green")
        else:
            p.print_v(f"{negative_rows.shape[0]}/{data.shape[0]} ({percentage_negative}%) "
                      f"negative values for fid_mark {fid_dir}", color="red")

            # show more details
            if debug_more_details:
                p.print_v(negative_ids)

        # calculate mean and std for the data
        mean_x = data[f"fid_mark_{fid_dir}_x"].mean()
        std_x = data[f"fid_mark_{fid_dir}_x"].std()
        mean_y = data[f"fid_mark_{fid_dir}_y"].mean()
        std_y = data[f"fid_mark_{fid_dir}_y"].std()

        # we allow a small buffer in the values
        buffer_range = 150
        min_range_x = mean_x - std_x - buffer_range
        max_range_x = mean_x + std_x + buffer_range
        min_range_y = mean_y - std_y - buffer_range
        max_range_y = mean_y + std_y + buffer_range

        # calculate how many values are outside the adapted std
        x_off = data[(data[f"fid_mark_{fid_dir}_x"] < min_range_x) |
                     (data[f"fid_mark_{fid_dir}_x"] > max_range_x)]
        y_off = data[(data[f"fid_mark_{fid_dir}_y"] < min_range_y) |
                     (data[f"fid_mark_{fid_dir}_y"] > max_range_y)]
        off_ids_x = x_off['image_id'].tolist()
        off_ids_y = y_off['image_id'].tolist()
        off_ids = off_ids_x + list(set(off_ids_y) - set(off_ids_x))

        # get these values also as percentages
        percentage_off_x = round(x_off.shape[0] / data.shape[0] * 100, 2)
        percentage_off_y = round(y_off.shape[0] / data.shape[0] * 100, 2)

        # check if we are good or not
        if percentage_off_x > 5 or percentage_off_y > 5:
            col = "red"
        else:
            col = "green"

        # print our results
        p.print_v(f"{x_off.shape[0]}/{data.shape[0]} ({percentage_off_x}%) x-values "
                  f"of fid_mark {fid_dir} are off, "
                  f"{y_off.shape[0]}/{data.shape[0]} ({percentage_off_y}%) y-values "
                  f"of fid_mark {fid_dir} are off", color=col)

        # if wished give more details
        if debug_more_details:
            if len(off_ids) > debug_max_cutoff:
                p.print_v(off_ids[0:debug_max_cutoff])
                p.print_v(f"list was cut off after {debug_max_cutoff} entries")
            else:
                p.print_v(off_ids)


if __name__ == "__main__":
    quality_check_table_fid_points()
