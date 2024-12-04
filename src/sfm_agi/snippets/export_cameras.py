"""Exports camera positions and heights"""

# Library imports
import pandas as pd

# local imports
import src.base.connect_to_database as ctd


def export_cameras(image_ids: list[str], csv_path: str,
                   extension="tif", conn=None) -> None:
    """
    Exports camera positions and heights to a CSV file that can be used in Agisoft Metashape.

    Args:
        image_ids (List[str]): List of image IDs to export.
        csv_path (str): Path to the CSV file where data will be exported.

    Returns:
        None
    """

    if conn is None:
        conn = ctd.establish_connection()

    # convert list of image ids to one string
    str_image_ids = "'" + "', '".join(image_ids) + "'"

    sql_string_base = f"SELECT image_id, altitude FROM images WHERE image_id IN ({str_image_ids})"
    data_base = ctd.execute_sql(sql_string_base, conn)

    sql_string_extracted = f"SELECT image_id, ST_AsText(position_exact)AS position_exact, height " \
                           f"FROM images_extracted WHERE image_id IN ({str_image_ids})"
    data_extracted = ctd.execute_sql(sql_string_extracted, conn)

    data = data_base.merge(data_extracted, on="image_id")

    # add extension to image_id
    data['image_id'] = data['image_id'] + "." + extension

    # set the heigth
    # data['height'] = 4998

    # split position_exact into x, y
    data['x'] = data['position_exact'].str.split(" ").str[0].str[6:].astype(float)
    data['y'] = data['position_exact'].str.split(" ").str[1].str[:-1].astype(float)

    def get_z(row):
        feet_to_meters = 0.3048  # Conversion factor from feet to meters
        if pd.notnull(row['height']):
            return row['height'] * feet_to_meters
        elif pd.notnull(row['altimeter_value']):
            return row['altimeter_value'] * feet_to_meters
        elif pd.notnull(row['altitude']) and row['altitude'] != -99999:
            return row['altitude'] * feet_to_meters
        return 22000 * feet_to_meters

    # Apply the function to get z values
    data['z'] = data.apply(get_z, axis=1)

    # round to 3 decimals
    data['x'] = data['x'].round(3)
    data['y'] = data['y'].round(3)
    data['z'] = data['z'].round(3)

    columns = ["image_id", "x", "y", "z"]

    add_rotation = True
    if add_rotation:

        data['yaw'] = 339.14
        data['pitch'] = 0
        data['roll'] = 0

        columns.append("yaw")
        columns.append("pitch")
        columns.append("roll")

    # export to csv
    data[columns].to_csv(csv_path, index=False)
