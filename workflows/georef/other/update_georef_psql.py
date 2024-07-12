# Library imports
import glob
import os
import numpy as np
import shapely
from shapely.geometry import LineString
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.georef.snippets.calc_azimuth as ca
import src.georef.snippets.calc_camera_position as ccp
import src.georef.snippets.convert_image_to_footprint as citf
import src.load.load_image as li

# Variables
input_fld = "/data/ATM/data_1/georef/"
methods = ["sat"]
overwrite = True


def update_georef_psql():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all data from the table
    sql_string = "SELECT * FROM images_georef_3"
    data = ctd.execute_sql(sql_string, conn)

    # get the approx positions
    sql_string = "SELECT image_id, ST_AsText(position_approx) AS position_approx FROM images_extracted"
    data_approx = ctd.execute_sql(sql_string, conn)

    for method in methods:

        fld = os.path.join(input_fld, method)

        # get all tiff files in the folder
        tif_files = glob.glob(os.path.join(fld, '*.tif'))

        # save how many entries are updated
        updated_entries = 0

        # iterate over all files
        for file in (pbar := tqdm(tif_files)):

            # get the file name without the extension
            image_id = os.path.basename(file)[:-4]

            # get the row with the image_id from the psql data
            row = data[data["image_id"] == image_id]
            row_approx = data_approx[data_approx["image_id"] == image_id]

            # check if image does already have a position
            if row.empty:
                # empty rows always work
                pass
            else:

                # check if the method is better
                better_method = False
                if method == "sat" and row["georef_type"].iloc[0] in ["img", "calc"]:
                    better_method = True
                if method == "img" and row["georef_type"].iloc[0] == "calc":
                    better_method = True

                if better_method:
                    # if the method is better, update
                    pass
                else:
                    if overwrite:
                        # we still update if overwrite is True
                        pass
                    else:
                        # we skip if the method is not better and overwrite is False
                        continue

            # update the progress bar
            pbar.set_postfix_str(f"Save position for {image_id} "
                                 f"({updated_entries} already updated)")

            # load the image, transform & points
            image = li.load_image(fld + "/" + image_id + ".tif")
            transform = np.loadtxt(fld + "/" + image_id + "_transform.txt")
            points = np.loadtxt(fld + "/" + image_id + "_points.txt")

            # get the footprint of the image
            footprint = citf.convert_image_to_footprint(image, transform)

            # get the camera center from the image
            camera_pos = ccp.calc_camera_position(footprint)

            # get nr of points and avg residuals
            nr_points = points.shape[0]
            try:
                avg_residuals = np.mean(points[:, 5])
            except:  # noqa
                avg_residuals = "Null"
            try:
                avg_points_quality = np.mean(points[:, 4])
            except:  # noqa
                avg_points_quality = "Null"

            azimuth = ca.calc_azimuth(image_id, conn)

            # calculate error vector
            if row_approx['position_approx'].iloc[0] is not None:
                approx_pos = shapely.from_wkt(row_approx["position_approx"]).iloc[0]
                error_vector = LineString([(camera_pos.x, camera_pos.y), (approx_pos.x, approx_pos.y)])
            else:
                error_vector = None

            # create sql_string (update or insert)
            if row.empty is False:
                sql_string = f"UPDATE images_georef_3 SET " \
                             f"georef_type='{method}', " \
                             f"footprint_exact=ST_GeomFromText('{footprint.wkt}'), " \
                             f"position_exact=ST_GeomFromText('{camera_pos}'), " \
                             f"nr_tps={nr_points}, " \
                             f"avg_residuals={avg_residuals}, " \
                             f"avg_points_quality={avg_points_quality}"

                if error_vector is not None:
                    sql_string += f", error_vector=ST_GeomFromText('{error_vector}')"
                if azimuth is not None:
                    sql_string += f", azimuth_exact={azimuth}"

                sql_string = sql_string + f" WHERE image_id='{image_id}'"
            else:
                # Insert new entry
                sql_string = f"INSERT INTO images_georef_3 (" \
                             f"image_id, georef_type, footprint_exact, " \
                             f"position_exact, nr_tps, avg_residuals, avg_points_quality"

                if error_vector is not None:
                    sql_string += ", error_vector"
                if azimuth is not None:
                    sql_string += ", azimuth_exact"

                sql_string += ") VALUES ("
                sql_string += f"'{image_id}', '{method}', ST_GeomFromText('{footprint.wkt}'), " \
                              f"ST_GeomFromText('{camera_pos}'), {nr_points}, " \
                              f"{avg_residuals}, {avg_points_quality}"

                if error_vector is not None:
                    sql_string += f", ST_GeomFromText('{error_vector}')"
                if azimuth is not None:
                    sql_string += f", {azimuth}"

                sql_string += ")"

            ctd.execute_sql(sql_string, conn)

            updated_entries += 1


if __name__ == "__main__":
    update_georef_psql()
