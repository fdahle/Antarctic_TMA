"""Creates a CSV file defining the camera positions of the images for use in MicMac"""

# Library imports
import csv
import os.path
from pyproj import Transformer
from typing import Union

# Local imports
import src.base.connect_to_database as ctd

# Variables
overwrite = True


def create_camera_csv(image_ids: list[Union[int, str]], csv_path: str, prefix: str = "",
                      input_epsg: int = 3031, output_epsg: int = 3031,
                      skip_missing: bool = False,
                      default_height: float = 5000.0) -> None:
    """
    Creates a CSV file defining the camera positions of the images for use in MicMac for OriConvert.
    Args:
        image_ids (List[Union[int, str]]): A list of the ids of the images that should be included
            in the csv.
        csv_path (str): The path to the CSV file to be created.
        prefix (str, optional): A prefix to be added to the image IDs in the CSV file. Defaults to "".
        input_epsg (int, optional): The EPSG code of the input coordinate system. Defaults to 3031.
        output_epsg (int, optional): The EPSG code of the output coordinate system. Defaults to 3031.
        skip_missing (bool, optional): Whether to skip images with missing height information.
            Defaults to False.
        default_height (float, optional): The default height to be used if height information
            is missing. Defaults to 5000.0.
    Returns:
        None
    Raises:
        FileExistsError: If the CSV file already exists and overwrite is set to False.
        ValueError: If the camera type is not found in the image ID.
    """

    # check if the file already exists
    if os.path.isfile(csv_path) and overwrite is False:
        raise FileExistsError(f"File {csv_path} already exists")

    # establish connection to psql
    conn = ctd.establish_connection()

    # convert lst of image_ids to string for sql query
    str_image_ids = ",".join(["'" + str(image_id) + "'" for image_id in image_ids])

    # get all camera parameters from the database
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
