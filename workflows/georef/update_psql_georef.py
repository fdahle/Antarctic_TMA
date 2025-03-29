# Library imports
import geopandas as gpd
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

# Local imports
import src.base.connect_to_database as ctd

# Variables
input_fld = "/data/ATM/data_1/georef"
overwrite_sat = True
overwrite_img = True
overwrite_calc = True
skip_invalid = True

flight_paths = [1833] # if None, all flight paths are used

update_types = ["sat", "img", "calc"]


def update_psql_georef():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get existing image ids from the database as list
    sql_string = "SELECT image_id FROM images_georef"
    existing_image_ids = ctd.execute_sql(sql_string, conn)

    if flight_paths is not None:
        str_fp = [str(fp) for fp in flight_paths]
        existing_image_ids = existing_image_ids[existing_image_ids["image_id"].str[2:6].isin(str_fp)]

    existing_image_ids = existing_image_ids["image_id"].tolist()

    for update_type in update_types:

        path_footprint_shp = input_fld + f"/footprints/{update_type}_footprints.shp"
        path_oblique_footprint_shp = input_fld + f"/footprints/{update_type}_footprints_oblique.shp"

        # load the footprints from pandas
        footprints = gpd.read_file(path_footprint_shp)
        footprints_oblique = gpd.read_file(path_oblique_footprint_shp)

        if flight_paths is not None:
            str_fp = [str(fp) for fp in flight_paths]
            footprints = footprints[footprints["image_id"].str[2:6].isin(str_fp)]
            footprints_oblique = footprints_oblique[footprints_oblique["image_id"].str[2:6].isin(str_fp)]

        # merge the footprints
        footprints_merged = gpd.GeoDataFrame(pd.concat([footprints, footprints_oblique], ignore_index=True))

        # iterate the file
        for i, row in tqdm(footprints_merged.iterrows(), total=footprints_merged.shape[0]):

            # get the image  id
            image_id = row["image_id"]

            # get the footprint
            footprint = row["geometry"]
            footprint_wkt = footprint.wkt

            if "32V" in image_id:
                # get the path to the points.txt file
                path_points_txt = input_fld + f"/images/{update_type}/{image_id}_points.txt"

                # check if the file exists
                if not os.path.exists(path_points_txt):
                    if skip_invalid:
                        print(f"File {path_points_txt} does not exist")
                        continue
                    else:
                        raise FileNotFoundError(f"File {path_points_txt} does not exist")

                # load the points.txt file
                points = np.loadtxt(path_points_txt)

                # load the points data
                nr_tps = points.shape[0]
                avg_points_quality = np.mean(points[:, 4])
                avg_residuals = np.mean(points[:, 5])

                # get the path to the transform.txt file
                path_transform_txt = input_fld + f"/images/{update_type}/{image_id}_transform.txt"

                # check if the file exists
                if not os.path.exists(path_transform_txt):
                    if skip_invalid:
                        print(f"File {path_transform_txt} does not exist")
                        continue
                    else:
                        raise FileNotFoundError(f"File {path_transform_txt} does not exist")

                # load the transform.txt file
                transform = np.loadtxt(path_transform_txt)

                # convert to string
                transform_string = ";".join(
                    [
                        ";".join([f"{value:.10g}" for value in row])  # Use .10g to remove trailing zeros
                        for row in transform
                    ]
                )

            if image_id not in existing_image_ids:
                if "32V" in image_id:
                    # create the sql string
                    sql_string = (f"INSERT INTO images_georef (image_id, "
                                    f"georef_type, "
                                    f"nr_tps, "
                                    f"avg_points_quality, "
                                    f"avg_residuals, "
                                    f"transform, "
                                    f"footprint_exact) "
                                    f"VALUES ('{image_id}', "
                                    f"'{update_type}', "
                                    f"{nr_tps}, "
                                    f"{avg_points_quality}, "
                                    f"{avg_residuals}, "
                                    f"'{transform_string}', "
                                    f"ST_GeomFromText('{footprint_wkt}', 3031))")
                else:
                    # create the sql string
                    sql_string = (f"INSERT INTO images_georef (image_id, "
                                    f"georef_type, "
                                    f"footprint_exact) "
                                    f"VALUES ('{image_id}', "
                                    f"'{update_type}', "
                                    f"ST_GeomFromText('{footprint_wkt}', 3031))")
            else:
                if "32V" in image_id:
                    # create the sql string
                    sql_string = (f"UPDATE images_georef SET "
                                  f"georef_type='{update_type}', "
                                  f"nr_tps={nr_tps}, "
                                  f"avg_points_quality={avg_points_quality}, "
                                  f"avg_residuals={avg_residuals}, "
                                  f"transform='{transform_string}', "
                                  f"footprint_exact=ST_GeomFromText('{footprint_wkt}', 3031) "
                                  f"WHERE image_id = '{image_id}'")
                else:
                    # create the sql string
                    sql_string = (f"UPDATE images_georef SET "
                                  f"georef_type='{update_type}', "
                                  f"footprint_exact=ST_GeomFromText('{footprint_wkt}', 3031) "
                                  f"WHERE image_id = '{image_id}'")

                # Apply additional conditions based on `update_type`
                if update_type == "sat":
                    # "sat" can overwrite anything unless `overwrite` is False
                    if overwrite_sat is False:
                        sql_string += " AND (footprint_exact IS NULL OR georef_type != 'sat')"
                elif update_type == "img":
                    # always skip 'sat' footprints
                    sql_string += " AND georef_type != 'sat'"
                    if overwrite_img is False:
                        # Avoid overwriting existing "img" footprints
                        sql_string += " AND (footprint_exact IS NULL OR georef_type != 'img')"
                elif update_type == "calc":
                    # always skip 'sat' and 'img' footprints
                    sql_string += " AND georef_type != 'sat' AND georef_type != 'img'"
                    if overwrite_calc is False:
                        # Avoid overwriting existing "calc" footprints
                        sql_string += " AND footprint_exact IS NULL"

            try:
                print(sql_string)
                ctd.execute_sql(sql_string, conn)
            except:
                print(f"Error with image_id: {image_id}")
                continue

if __name__ == "__main__":
    update_psql_georef()
