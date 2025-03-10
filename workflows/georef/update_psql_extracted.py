import src.base.connect_to_database as ctd

def update_psql_extracted():

    conn = ctd.establish_connection()

    # sql string for vertical images
    sql_string = ("UPDATE images_extracted AS ie "
                  "SET footprint_type = ig.georef_type, "
                  "azimuth_exact = ig.azimuth_exact, "
                  "footprint_exact = ig.footprint_exact,"
                  "position_exact = ig.position_exact, "
                  "error_vector = ig.error_vector "
                  "FROM images_georef AS ig "
                  "WHERE ie.image_id = ig.image_id AND "
                  "ig.georef_type != 'failed'")
    ctd.execute_sql(sql_string, conn, add_timestamp=False)

    # sql string for oblique images
    sql_string = (
        "UPDATE images_extracted AS ie "
        "SET footprint_type = ig.georef_type, "
        "    azimuth_exact = ig.azimuth_exact, "
        "    position_exact = ig.position_exact, "
        "    error_vector = ig.error_vector "
        "FROM images_georef AS ig "
        "WHERE (ie.image_id = ig.image_id "
        "       OR ie.image_id = regexp_replace(ig.image_id, '32V', '31L', 'g') "
        "       OR ie.image_id = regexp_replace(ig.image_id, '32V', '33R', 'g')) "
        "  AND ig.georef_type != 'failed'"
    )
    ctd.execute_sql(sql_string, conn, add_timestamp=False)

if __name__ == "__main__":
    update_psql_extracted()