# Library imports
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.load.load_image as li
import src.prepare.check_sky as cs

# Constants
PATH_SEGMENTED_FLD = "/data/ATM/data_1/aerial/TMA/segmented/unet/"

# Variables
overwrite = True


def check_sky_for_images():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get images and sky from the database
    sql_string = "SELECT image_id, sky_is_correct FROM images"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # filter out the images that already have a footprint
    if overwrite is False:
        data = data[data['sky_is_correct'].isnull()]

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Check sky for {image_id} "
                             f"({updated_entries} already updated)")

        path_segmented_image = PATH_SEGMENTED_FLD + image_id + ".tif"

        # load the segmented image
        try:
            image = li.load_image(path_segmented_image)
        except (Exception,):
            continue

        # check if the sky is correct
        sky_is_correct = cs.check_sky(image)

        if sky_is_correct is None:
            continue

        # create sql string
        sql_string = f"UPDATE images SET sky_is_correct={sky_is_correct} " \
                     f"WHERE image_id='{image_id}'"

        # update the database
        ctd.execute_sql(sql_string, conn)

        # update the counter
        updated_entries += 1


if __name__ == "__main__":
    check_sky_for_images()
