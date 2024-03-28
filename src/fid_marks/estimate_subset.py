import src.base.connect_to_database as ctd


MIN_NR_OF_IMAGES = None
MAX_STD = None

def estimate_subset(image_id, key, conn=None):

    if conn is None:
        conn = ctd.establish_connection()

    # get the properties of this image (flight path, etc)
    sql_string = f"SELECT tma_number, view_direction, id_cam FROM images WHERE image_id='{image_id}'"
    data_image_properties = ctd.execute_sql(sql_string, conn)
