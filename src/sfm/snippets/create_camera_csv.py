# Package imports
import csv
import os.path

# Custom imports
import src.base.connect_to_database as ctd

# Variables
overwrite = False

def create_camera_csv(image_ids, csv_path, prefix="", skip_missing=False):

    if os.path.isfile(csv_path) and overwrite is False:
        raise FileExistsError(f"File {csv_path} already exists")

    # establish connection to psql
    conn = ctd.establish_connection()

    # convert lst of image_ids to string for sql query
    str_image_ids = ",".join(["'" + str(image_id) + "'" for image_id in image_ids])

    # get all cameria parameters from the database
    sql_string = "SELECT images.image_id, " \
                 "ST_AsText(position_exact) AS position_exact, " \
                 "azimuth_exact, height " \
                 "FROM images " \
                 "JOIN images_extracted ON images.image_id = images_extracted.image_id " \
                 "WHERE images.image_id IN (" + str_image_ids + ")"
    data = ctd.execute_sql(sql_string, conn)

    # add a prefix to the image_ids
    if len(prefix) > 0:
        data['image_id'] = prefix + data['image_id']

    # get x and y coordinates
    data['x'] = [point.split(" ")[0][6:] for point in data['position_exact']]
    data['y'] = [point.split(" ")[1][:-1] for point in data['position_exact']]
    data = data.drop(columns=['position_exact'])

    print(data)

    exit()



    cameras = []


    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)

        # write the header rows
        writer.writerow("#F=N Y X Z K W P")
        writer.writerow("#")

        # loop over pandas dataframe
        for idx, row in data.iterrows():

            # create the csv string
            csv_row = f"{row['image_id']} {row['x']} {row['y']} " \
                      f"{row['height']} {row['azimuth_exact']} 0 0"

            # write the camera parameters
            writer.writerow(camera)
