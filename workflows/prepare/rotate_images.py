# Library imports
import os
import pandas as pd
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.load.load_image as li
import src.prepare.correct_image_orientation as cio

# Display imports
import src.display.display_images as di

# Constants
PATH_IMAGE_FLD = "/data/ATM/data_1/aerial/TMA/downloaded/"


# Debug parameters
debug_display_images = False


image_ids = ["CA173632V0009"]

def rotate_images(image_ids=None):

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images with their rotation
    sql_string = ("SELECT image_id, "
                  "rotation_sidebar_corrected FROM images")

    # add constraint for image ids
    if image_ids is not None:
        sql_string += f" WHERE image_id IN {tuple(image_ids)}"

    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0
    rotated_entries = 0

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # skip if image is already checked for rotation
        #if pd.notnull(row['rotation_sidebar_corrected']):
        #    continue

        pbar.set_postfix_str(f"Check rotation for {row['image_id']} "
                             f"({rotated_entries}/{updated_entries} rotated)")

        # set path to image
        path_img = os.path.join(PATH_IMAGE_FLD, row['image_id'] + ".tif")

        # load the image
        try:
            image = li.load_image(row['image_id'])
        except (Exception,):
            continue

        # (possibly rotate the image)
        rotation_was_required = cio.correct_image_orientation(image,
                                                              path_img)

        if rotation_was_required:
            rotated_entries += 1

        if debug_display_images and rotation_was_required:
            di.display_images(image)

        # update the database
        sql_string = f"UPDATE images SET rotation_sidebar_corrected=True WHERE image_id='{row['image_id']}'"
        ctd.execute_sql(sql_string, conn)

        updated_entries += 1


if __name__ == "__main__":
    rotate_images(image_ids)
