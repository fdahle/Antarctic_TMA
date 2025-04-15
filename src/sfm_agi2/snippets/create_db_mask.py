import numpy as np

import src.base.connect_to_database as ctd

def create_db_mask(mask_shape, image_id, conn=None):

    if conn is None:
        conn = ctd.establish_connection()

    # create a placeholder mask that we fill with the text boxes
    mask = np.ones(mask_shape, dtype=np.bool_)

    # get the text boxes for that image
    sql_string = f"SELECT text_bbox FROM images_extracted WHERE image_id ='{image_id}'"
    data = ctd.execute_sql(sql_string, conn)

    text_string = data.iloc[0]["text_bbox"]
    if len(text_string) > 0 and "[" not in text_string:
        text_string = "[" + text_string + "]"

    # Replace square brackets with empty strings and semicolons with commas
    text_string = text_string.replace("[", "").replace("]", "").replace(";", ",")

    # replace normal brackets with empty strings
    text_string = text_string.replace("(", "").replace(")", "")

    # Split the string into individual elements by commas
    elements = text_string.split(",")

    # Group the elements in chunks of 4 and convert them into tuples of four integers
    text_boxes = [(int(float(elements[i])), int(float(elements[i + 1])),
                   int(float(elements[i + 2])), int(float(elements[i + 3])))
                  for i in range(0, len(elements), 4)  # Step by 4 to create tuples of 4 ints
                  ]

    # Set those regions in the mask to False (0)
    for x1, y1, x2, y2 in text_boxes:
        mask[y1:y2, x1:x2] = False

    return mask, conn