# Package imports
import os
import numpy as np
import shapely
from shapely.geometry import LineString
from tqdm import tqdm

# Custom imports
import src.base.connect_to_database as ctd
import src.georef.snippets.calc_camera_position as ccp
import src.georef.snippets.convert_image_to_footprint as citf
import src.load.load_image as li

# Variables
input_fld = "/data_1/ATM/data_1/georef/"
method = "sat"
overwrite = True


def update_georef_psql():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get the georef data from psql
    sql_string = "SELECT image_id, footprint_type, footprint_exact, position_exact, " \
                 "ST_AsText(position_approx) AS position_approx " \
                 "FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    georef_fld = input_fld + "/" + method

    # get all tif files in the fld
    image_ids = [file[:-4] for file in os.listdir(georef_fld) if file.endswith(".tif")]

    # save how many entries are updated
    updated_entries = 0

    # iterate all images in the fld
    for image_id in (pbar := tqdm(image_ids)):

        pbar.set_postfix_str(f"Save position for {image_id} "
                             f"({updated_entries} already updated)")


        # get the row with the image_id from the psql data
        row = data[data["image_id"] == image_id]

        # check if we should overwrite the data
        if overwrite is False and \
                row['footprint_exact'] is not None and \
                row['position_exact'] is not None:
            continue

        # don't overwrite more accurate method
        if method == "sat":
            pass
        elif method == "img" and row['footprint_type'] == "sat":
            continue
        elif method == "calc" and row['footprint_type'] in ["sat", "img"]:
            continue

        # load the image
        image = li.load_image(input_fld + "/" + method + "/" + image_id + ".tif")

        # get the transform
        transform = np.loadtxt(input_fld + "/" + method + "/" + image_id + "_transform.txt")

        # get the footprint of the image
        footprint = citf.convert_image_to_footprint(image, transform)

        # get the camera center from the image
        camera_pos = ccp.calc_camera_position(footprint)

        # get the error vector
        if row["position_approx"].iloc[0] is not None:

            approx_pos = shapely.from_wkt(row["position_approx"].iloc[0])

            error_vector = LineString([(camera_pos.x, camera_pos.y), (approx_pos.x, approx_pos.y)])
        else:
            error_vector = None

        # create sql_string
        sql_string = f"UPDATE images_extracted SET " \
                     f"footprint_exact=ST_GeomFromText('{footprint.wkt}'), " \
                     f"position_exact=ST_GeomFromText('{camera_pos}'), " \
                     f"footprint_type='{method}' "
        if error_vector is not None:
            sql_string += f", position_error_vector=ST_GeomFromText('{error_vector}') " \

        sql_string = sql_string + f"WHERE image_id='{image_id}'"
        ctd.execute_sql(sql_string, conn)

        updated_entries += 1

if __name__ == "__main__":
    update_georef_psql()

