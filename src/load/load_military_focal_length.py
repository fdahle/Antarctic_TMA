import src.base.connect_to_database as ctd
def load_military_focal_length(image_id, cam_id=None, conn=None):

    # establish connection if not provided
    if conn is None:
        conn = ctd.establish_connection()

    # get the camera of the image
    if cam_id is None:
        sql_string = f"SELECT cam_id FROM images_extracted WHERE image_id = '{image_id}'"
        data = ctd.execute_sql(sql_string, conn)
        cam_id = data['cam_id'].iloc[0]

    # without a cam_id we cannot continue
    if cam_id is None:
        return None

    # get the focal length
    sql_string = f"SELECT cfl FROM military_calibration WHERE camera = '{cam_id}'"
    data = ctd.execute_sql(sql_string, conn)

    focal_length = data['cfl'].iloc[0]

    return focal_length