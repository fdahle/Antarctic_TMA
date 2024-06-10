import src.base.connect_to_database as ctd

csv_path = "/home/fdahle/SFTP/staff-umbrella/ATM/images_orig/cameras.csv"


def export_cameras(image_ids):
    conn = ctd.establish_connection()

    # convert list of image ids to one string
    str_image_ids = "'" + "', '".join(image_ids) + "'"

    sql_string_base = f"SELECT image_id, altitude FROM images WHERE image_id IN ({str_image_ids})"
    data_base = ctd.execute_sql(sql_string_base, conn)

    sql_string_extracted = f"SELECT image_id, ST_AsText(position_exact)AS position_exact, height " \
                           f"FROM images_extracted WHERE image_id IN ({str_image_ids})"
    data_extracted = ctd.execute_sql(sql_string_extracted, conn)

    data = data_base.merge(data_extracted, on="image_id")

    data['height'] = 4998

    # split position_exact into x, y
    data['x'] = data['position_exact'].str.split(" ").str[0].str[6:].astype(float)
    data['y'] = data['position_exact'].str.split(" ").str[1].str[:-1].astype(float)

    # export to csv
    data[["image_id", "x", "y", "height"]].to_csv(csv_path, index=False)
