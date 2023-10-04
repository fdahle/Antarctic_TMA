import base.connect_to_db as ctd
import base.print_v as p

debug_more_details = True


def quality_check_table_segmentation():

    # get data from table
    sql_string = f"SELECT * FROM images_segmentation"
    data = ctd.get_data_from_db(sql_string)

    # count the number of rows with at least one NaN value
    nan_rows_mask = data.isna().any(axis=1)
    nan_rows = data[nan_rows_mask]
    num_nan_rows = nan_rows_mask.sum()
    num_nan_percentage = round(num_nan_rows/data.shape[0] * 100, 2)

    if num_nan_rows > 0:
        p.print_v(f"For {num_nan_rows}/{data.shape[0]} ({num_nan_percentage}%) images "
                  f"the segmentation is incomplete", color="red")
        if debug_more_details:
            print(nan_rows['image_id'].tolist())
    else:
        p.print_v("For no images the segmentation is incomplete", color="green")

    # select only columns that start with "_perc"
    perc_cols = data.filter(regex=r'^perc_').columns

    # compute the sum of each row for the selected columns
    data['perc_sum'] = data[perc_cols].sum(axis=1)

    # count the number of rows that have a percentage of 100 (within 0.01)
    tolerance = 0.01
    mask = (data['perc_sum'].abs() - 100).round(2).abs() > tolerance
    non_100_rows = data[mask]
    num_non_100_rows = non_100_rows.shape[0]
    num_non_100_percentage = round(num_non_100_rows/data.shape[0] * 100, 2)

    if num_non_100_rows > 0:
        p.print_v(f"For {num_non_100_rows}/{data.shape[0]} ({num_non_100_percentage}%) images "
                  f"the segmentation is not adding up to 100% (with 1% error tolerance)",
                  color="red")
        if debug_more_details:
            print(non_100_rows['image_id'].tolist())
    else:
        p.print_v("For every image the segmentation adds up to 100% (with 1% error tolerance)",
                  color="green")


if __name__ == "__main__":

    quality_check_table_segmentation()
