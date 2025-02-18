import src.base.connect_to_database as ctd

flight_paths = [1352]
reset_cols = ["footprint_exact", "position_exact", "azimuth_exact", "error_vector", "nr_tps", "avg_residuals",
              "avg_points_quality", "transform"]

def set_georef_failed():

    # create connection to database
    conn = ctd.establish_connection()

    for flight_path in flight_paths:

        sql_string = "UPDATE images_georef SET georef_type='failed' WHERE SUBSTRING(image_id, 3, 4) = '" + str(flight_path) + "'"
        ctd.execute_sql(sql_string, conn)

        for col in reset_cols:

            sql_string = "UPDATE images_georef SET " + col + " = NULL WHERE SUBSTRING(image_id, 3, 4) = '" + str(flight_path) + "'"
            ctd.execute_sql(sql_string, conn)


if __name__ == "__main__":
    set_georef_failed()