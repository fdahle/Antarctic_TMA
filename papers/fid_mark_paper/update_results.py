import src.base.connect_to_database as ctd

def update_results():

    PATH_RESULTS = "results2.txt"

    # read the results
    with open(PATH_RESULTS, "r") as f:
        results = f.readlines()

    conn = ctd.establish_connection()

    # iterate the results
    for result in results:

        # split the result
        image_id, altitude = result.split(",")

        # remove the newline
        altitude = altitude.replace("\n", "")

        # update the database
        sql_string = (f"UPDATE images_extracted SET "
                      f"altimeter_value={altitude}, "
                      f"altimeter_estimated=False "
                      f"WHERE image_id='{image_id}'")

        print(sql_string)

        # execute the sql
        ctd.execute_sql(sql_string, conn)
    conn.close()

if __name__ == "__main__":
    update_results()