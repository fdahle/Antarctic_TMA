import pandas as pd
import src.base.connect_to_database as ctd


MIN_NR_IMAGES = 3
MIN_PERCENTAGE = 80

def clean_data(
        clean_focal_length: bool = True,
):

    # create connection to the database
    conn = ctd.establish_connection()

    sql_string = "SELECT * FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # create column for flight path
    data['flight_path'] = data['image_id'].str[2:9]

    if clean_focal_length:

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

        print(focal_length_percentage_dict)

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
