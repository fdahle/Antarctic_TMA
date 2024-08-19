import os

import src.base.connect_to_database as ctd
import src.base.rotate_image as ri
import src.export.export_tiff as et
import src.load.load_image as li

def check_sky(image_path, image_id=None, conn=None):

    # extract image_id from image_path if not given
    if image_id is None:
        image_id = os.path.basename(image_path).split(".")[0]

    if conn is None:
        conn = ctd.establish_connection()

    # get rotation info for the images
    sql_string = f"SELECT sky_is_correct FROM images WHERE image_id='{image_id}'"
    sky_data = ctd.execute_sql(sql_string, conn)
    sky_is_correct = sky_data["sky_is_correct"].values[0]
    sky_is_correct = bool(sky_is_correct)

    if sky_is_correct is False:

        # load the image
        img = li.load_image(image_path)

        # rotate the image
        img = ri.rotate_image(img, 180)

        # save the rotated image
        et.export_tiff(img, image_path, overwrite=True)

    return sky_is_correct, conn
