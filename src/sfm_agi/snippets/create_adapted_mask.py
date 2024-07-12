import copy
import cv2
import numpy as np

import src.base.connect_to_database as ctd
import src.base.create_mask as cm

def create_adapted_mask(existing_mask, image_id, conn=None):

    # copy the existing mask
    new_mask = copy.deepcopy(existing_mask)

    inverted_mask = cv2.bitwise_not(new_mask)

    kernel_size = 25
    iterations = 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_mask = cv2.dilate(inverted_mask, kernel, iterations=iterations)

    kerneled_mask = cv2.bitwise_not(dilated_mask)
    kerneled_mask = kerneled_mask.astype(np.uint8)
    kerneled_mask = kerneled_mask * 255

    # create a placeholder mask that we fill with the text boxes
    placeholder_mask = np.ones_like(new_mask)

    # create connection to the database if not provided
    if conn is None:
        conn = ctd.establish_connection()

    # get the text boxes for that image
    sql_string = f"SELECT text_bbox FROM images_extracted WHERE image_id ='{image_id}'"
    data = ctd.execute_sql(sql_string, conn)
    text_string = data.iloc[0]["text_bbox"]
    if len(text_string) > 0 and "[" not in text_string:
        text_string = "[" + text_string + "]"

    # create text-boxes list
    text_boxes = [list(group) for group in eval(text_string.replace(";", ","))]

    # create the mask with the text boxes
    text_mask = cm.create_mask(placeholder_mask, ignore_boxes=text_boxes,
                          use_default_fiducials=True, default_fid_position=0)

    text_mask = text_mask * 255

    mask = np.minimum(kerneled_mask, text_mask)

    mask = mask.astype(np.uint8)

    return mask
