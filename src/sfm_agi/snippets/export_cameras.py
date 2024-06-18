"""Exports camera positions and heights"""

# local imports
import src.base.connect_to_database as ctd


def export_cameras(image_ids: list[str], csv_path: str) -> None:
    """
    Exports camera positions and heights to a CSV file that can be used in Agisoft Metashape.

    Args:
        image_ids (List[str]): List of image IDs to export.
        csv_path (str): Path to the CSV file where data will be exported.

    Returns:
        None
    """
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
