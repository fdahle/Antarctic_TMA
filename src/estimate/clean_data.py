"""Make data in images_extracted more consistent by updating focal lengths and
heights based on the most common values."""

# Library imports
import pandas as pd
import src.base.connect_to_database as ctd

# Constants
MIN_NR_IMAGES = 3
MIN_PERCENTAGE = 80


def clean_focal_length() -> None:
    """
    Adjusts focal lengths in the images_extracted table in the database to ensure consistency.
    For each flight path, the focal length is updated to the most common focal length if it is present in more than
    MIN_PERCENTAGE percent of the images in the flight path. At least MIN_NR_IMAGES images are required to update the
    focal length.
    Returns:
        None
    """

    # create connection to the database
    conn = ctd.establish_connection()

    sql_string = "SELECT * FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # create column for flight path
    data['flight_path'] = data['image_id'].str[2:9]

    # get image_id and focal_length
    data_fl = data[['image_id', 'flight_path', 'focal_length']]

    # drop rows with missing focal length
    data_fl = data_fl.dropna()

    # group by flight path
    grouped_fl = data_fl.groupby('flight_path')

    # Create a dictionary with flight paths as keys and lists of focal lengths as values
    focal_length_dict = {flight_path: group['focal_length'].tolist() for flight_path, group in grouped_fl}

    # remove the entries where the focal length is the same for all images
    focal_length_dict = {flight_path: focal_lengths for flight_path, focal_lengths in focal_length_dict.items()
                         if len(set(focal_lengths)) > 1}

    # remove the entries where we have less than 3 focal lengths
    focal_length_dict = {flight_path: focal_lengths for flight_path, focal_lengths in focal_length_dict.items()
                         if len(focal_lengths) >= MIN_NR_IMAGES}

    # Calculate the percentage of each unique focal length for each flight path
    focal_length_percentage_dict = {}
    for flight_path, focal_lengths in focal_length_dict.items():
        total_count = len(focal_lengths)
        focal_length_counts = pd.Series(focal_lengths).value_counts()
        focal_length_percentages = ((focal_length_counts / total_count) * 100).round(2)
        focal_length_percentage_dict[flight_path] = focal_length_percentages.to_dict()

    # create copy of column focal_length
    data_fl['focal_length_new'] = data_fl['focal_length']

    # Update the focal lengths in the data based on the MIN_PERCENTAGE criterion
    for flight_path, focal_lengths in focal_length_dict.items():
        total_count = len(focal_lengths)
        focal_length_counts = pd.Series(focal_lengths).value_counts()
        focal_length_percentages = ((focal_length_counts / total_count) * 100).round(2)

        # Check if any focal length meets the MIN_PERCENTAGE criterion
        for focal_length, percentage in focal_length_percentages.items():
            if percentage >= MIN_PERCENTAGE:
                # Update all focal lengths in this group to the focal_length that meets the criterion
                data_fl.loc[data_fl['flight_path'] == flight_path, 'focal_length_new'] = focal_length
                break

    data_to_update = data_fl[data_fl['focal_length'] != data_fl['focal_length_new']]

    for index, row in data_to_update.iterrows():
        sql_string = f"UPDATE images_extracted SET focal_length={row['focal_length_new']} " \
                     f"WHERE image_id='{row['image_id']}';"
        ctd.execute_sql(sql_string, conn)


def clean_fl_cam_id(min_percentage: int = 50) -> None:
    """
    Adjusts focal lengths in the images_extracted table based on cam_id to ensure consistency. For each cam_id,
    the focal length is updated to the most common focal length if it is present in more than MIN_PERCENTAGE
    percent of the images with that cam_id.

    Args:
        min_percentage (int, optional): Minimum percentage of a focal length occurrence to be considered valid.
            Defaults to 50.
    Returns:
        None
    """
    # create connection to the database
    conn = ctd.establish_connection()

    sql_string = "SELECT image_id, cam_id, focal_length FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # get the unique focal lengths and counts for each cam_id
    grouped_count = data.groupby(['cam_id', 'focal_length']).size().reset_index(name='count')

    #  order by cam_id
    grouped_count = grouped_count.sort_values(by=['cam_id'], ascending=False)

    # get the unique cam_ids
    cam_ids = grouped_count['cam_id'].unique()

    # iterate over all cam_ids
    for cam_id in cam_ids:

        # get all focal lengths for this cam_id
        cam_data = grouped_count[grouped_count['cam_id'] == cam_id]
        cam_data = cam_data.reset_index(drop=True)
        if cam_data.shape[0] > 1 and cam_data['count'].max() / cam_data['count'].sum() > min_percentage / 100:
            print(cam_data)

            # get the index of the entry with the highest count
            max_count_index = cam_data['count'].idxmax()

            # get focal length of entry with the highest count
            focal_length = cam_data.loc[max_count_index, 'focal_length']

            sql_string = f"UPDATE images_extracted SET focal_length={focal_length} " \
                         f"WHERE cam_id='{cam_id}';"
            ctd.execute_sql(sql_string, conn)


def clean_height_altitude() -> None:
    """
        Adjusts heights in the images_extracted table in the database to ensure consistency.
        For each flight path, the heights is updated to the most common heights if it is present in more than
        MIN_PERCENTAGE percent of the images in the flight path. At least MIN_NR_IMAGES images are required to
        update the heights.
        Args:
        Returns:
            None
    """

    # create connection to the database
    conn = ctd.establish_connection()

    sql_string = "SELECT images_extracted.image_id, images_extracted.height, " \
                 "images_extracted.altimeter_value, images.altitude FROM images_extracted " \
                 "INNER JOIN images ON images_extracted.image_id = images.image_id;"
    data = ctd.execute_sql(sql_string, conn)

    # remove rows with missing height or altimeter_value
    data = data.dropna()

    print(data)
