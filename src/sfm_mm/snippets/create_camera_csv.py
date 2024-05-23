# Package imports
import csv
import os.path
from pyproj import Transformer

# Custom imports
import src.base.connect_to_database as ctd

# Variables
overwrite = True


def create_camera_csv(image_ids, csv_path, prefix="",
                      input_epsg=3031, output_epsg=3031,
                      skip_missing=False,
                      default_height=5000):

    # check if the file already exists
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
        data['image_id'] = prefix + data['image_id'] + ".tif"

    # Prepare transformer to convert coordinates
    transformer = Transformer.from_crs(f"EPSG:{input_epsg}", f"EPSG:{output_epsg}", always_xy=True)

    # get x and y coordinates and transform them
    data['x'], data['y'] = zip(*data['position_exact'].apply(
        lambda pos: transformer.transform(float(pos.split(" ")[0][6:]),
                                          float(pos.split(" ")[1][:-1]))))
    data = data.drop(columns=['position_exact'])

    print(data)

    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ', quoting=csv.QUOTE_MINIMAL)

        # write the header rows
        writer.writerow(['#F=N', 'Y', 'X', 'Z', 'K', 'W', 'P'])
        writer.writerow(['#'])

        # loop over pandas dataframe
        for idx, row in data.iterrows():

            # some values are dependent on the camera type
            if "V" in row['image_id']:
                pitch = 90.0
                roll = 0.0
            elif "R" in row['image_id']:
                pass
            elif "L" in row['image_id']:
                pass
            else:
                raise ValueError(f"Camera type not found in {row['image_id']}")

            if skip_missing and row['height'] is None:
                continue

            if row['height'] is None:
                row['height'] = default_height

            # create the csv row
            csv_row = [row['image_id'], row['x'], row['y'], row['height'],
                       row['azimuth_exact'], pitch, roll]

            # write the camera parameters
            writer.writerow(csv_row)
