import numpy as np

import src.base.connect_to_database as ctd
import src.base.create_mask as cm

def create_text_mask(width, height, image_id, conn=None):

    placeholder_mask = np.ones((height, width))

    if conn is None:
        conn = ctd.establish_connection()

    sql_string = f"SELECT text_bbox FROM images_extracted WHERE image_id ='{image_id}'"
    data = ctd.execute_sql(sql_string, conn)

    # get the text boxes of the image
    text_string = data.iloc[0]["text_bbox"]

    if len(text_string) > 0 and "[" not in text_string:
        text_string = "[" + text_string + "]"

    # create text-boxes list
    text_boxes = [list(group) for group in eval(text_string.replace(";", ","))]

    mask = cm.create_mask(placeholder_mask, ignore_boxes=text_boxes,
                          use_default_fiducials=True, default_fid_position=0)

    mask = mask * 255

    return mask


if __name__ == "__main__":

    _img_id = "CA196532V0010"
    import src.load.load_image as li
    _img = li.load_image(_img_id)

    _mask = np.zeros_like(_img)

    _new_mask = create_text_mask(_mask, _img_id)

    import src.display.display_images as di
    di.display_images(_new_mask)