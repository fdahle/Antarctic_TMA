from rasterio.windows import shape

import src.base.connect_to_database as ctd
import src.load.load_shape_data as lsd


bool_update_table_extracted = False
bool_update_table_georef = False
bool_update_shape_files = True

flight_paths = [1352,1353,1354,1355,1356,1357,1358,1680,1681,1682,1683,1684,
1800,1803,1804,1809,1810,1811,1814,1819,1820,1822,1825,1826,
1827,1828,1830,1831,1832,1834,1844,1962,1967,1969,2075,2135,
2140,2159,2161,2162,2165,2167,1741,1963,1748]

path_shape_files = "/data/ATM/data_1/georef/footprints"

def set_georef_failed():

    # create connection to database
    conn = ctd.establish_connection()

    if bool_update_table_extracted:
        # reset columns for images_extracted
        reset_cols = ["footprint_exact", "position_exact", "azimuth_exact", "error_vector"]
        for flight_path in flight_paths:

            sql_string = "UPDATE images_extracted SET footprint_type='failed' WHERE SUBSTRING(image_id, 3, 4) = '" + str(flight_path) + "'"
            ctd.execute_sql(sql_string, conn)

            for col in reset_cols:

                sql_string = "UPDATE images_extracted SET " + col + " = NULL WHERE SUBSTRING(image_id, 3, 4) = '" + str(flight_path) + "'"
                ctd.execute_sql(sql_string, conn)

    if bool_update_table_georef:
        # reset columns for images_georef
        reset_cols = ["footprint_exact", "position_exact", "azimuth_exact", "error_vector", "nr_tps", "avg_residuals",
                      "avg_points_quality", "transform"]

        for flight_path in flight_paths:

            sql_string = "UPDATE images_georef SET georef_type='failed' WHERE SUBSTRING(image_id, 3, 4) = '" + str(flight_path) + "'"
            ctd.execute_sql(sql_string, conn)

            for col in reset_cols:

                sql_string = "UPDATE images_georef SET " + col + " = NULL WHERE SUBSTRING(image_id, 3, 4) = '" + str(flight_path) + "'"
                ctd.execute_sql(sql_string, conn)

    if bool_update_shape_files:
        # remove entries from shape files
        for gtype in ["sat", "img", "calc"]:

            path_shp = path_shape_files + "/" + gtype + "_footprints.shp"
            path_shp_oblique = path_shape_files + "/" + gtype + "_footprints_oblique.shp"

            # load shapedata
            shape_data = lsd.load_shape_data(path_shp)
            shape_data_oblique = lsd.load_shape_data(path_shp_oblique)

            # remove entries
            for flight_path in flight_paths:
                shape_data = shape_data[shape_data["image_id"].str[2:6] != str(flight_path)]
                shape_data_oblique = shape_data_oblique[shape_data_oblique["image_id"].str[2:6] != str(flight_path)]

            shape_data.to_file(path_shp)
            shape_data_oblique.to_file(path_shp_oblique)


if __name__ == "__main__":
    set_georef_failed()