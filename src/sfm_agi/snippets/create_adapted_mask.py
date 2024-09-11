"""adapt a agisoft metashape mask"""

# Library imports
import copy
import cv2
import numpy as np
import psycopg2

# Local imports
import src.base.connect_to_database as ctd
import src.base.create_mask as cm
import src.base.rotate_image as ri


def create_adapted_mask(existing_mask: np.ndarray, image_id: str,
                        rotated: bool = False,
                        conn: psycopg2.extensions.connection | None = None):
    """
    Adapt an existing mask from agisoft by adding the text boxes from the database.
    Furthermore, the mask is increased at all sizes to mask the borders as well
    (as agisoft create masks just too small). If the image is rotated extra code
    is necessary to have the text boxes in the correct position.
    Args:
        existing_mask (np.array): The existing mask from agisoft
        image_id (str): The image id of the image
        rotated (bool): Whether the image is rotated or not
        conn (pyodbc.Connection): An connection object to the database. Defaults to None.

    Returns:
        mask (np.array): The adapted mask

    """
    # create connection to the database if not provided
    if conn is None:
        conn = ctd.establish_connection()

    # copy the existing mask
    new_mask = copy.deepcopy(existing_mask)

    # rotate the mask back if necessary
    if rotated:
        new_mask = ri.rotate_image(new_mask, 180)

    inverted_mask = cv2.bitwise_not(new_mask)

    kernel_size = 25
    iterations = 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_mask = cv2.dilate(inverted_mask, kernel, iterations=iterations)

    kernel_mask = cv2.bitwise_not(dilated_mask)

    # create a placeholder mask that we fill with the text boxes
    placeholder_mask = np.ones_like(new_mask)

    # get the text boxes for that image
    sql_string = f"SELECT text_bbox FROM images_extracted WHERE image_id ='{image_id}'"
    data = ctd.execute_sql(sql_string, conn)
    text_string = data.iloc[0]["text_bbox"]
    if len(text_string) > 0 and "[" not in text_string:
        text_string = "[" + text_string + "]"

    print(text_string)

    # Replace square brackets with empty strings and semicolons with commas
    text_string = text_string.replace("[", "").replace("]", "").replace(";", ",")

    # replace normal brackets with empty strings
    text_string = text_string.replace("(", "").replace(")", "")

    # Split the string into individual elements by commas
    elements = text_string.split(",")

    # Group the elements in chunks of 4 and convert them into tuples of four integers
    text_boxes = [(int(elements[i]), int(elements[i + 1]), int(elements[i + 2]), int(elements[i + 3]))
                  for i in range(0, len(elements), 4)  # Step by 4 to create tuples of 4 ints
                  ]

    # create the mask with the text boxes
    text_mask = cm.create_mask(placeholder_mask, ignore_boxes=text_boxes,
                               use_default_fiducials=True, default_fid_position=0)

    mask = np.minimum(kernel_mask, text_mask)

    mask = mask.astype(np.uint8)

    # replace 1 with 255
    mask[mask == 1] = 255

    # rotate the mask back if necessary
    if rotated:
        mask = ri.rotate_image(mask, 180)

    return mask
