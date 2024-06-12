# Library imports
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.base.create_mask as cm
import src.load.load_image as li
import src.prepare.calc_complexity as cc

# variables
overwrite = False


def calc_complexities():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images and their complexity from the database
    sql_string = "SELECT image_id, complexity FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # filter out the images that already have a complexity
    if overwrite is False:
        data = data[data['complexity'].isnull()]

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Calculate complexity for {image_id} "
                             f"({updated_entries} already updated)")

        # load the image
        try:
            image = li.load_image(image_id)
        except (Exception,):
            continue

        # create mask for that image
        mask = cm.create_mask(image, use_default_fiducials=True)

        # calculate the complexity of the image
        complexity = cc.calc_complexity(image, mask=mask)

        if complexity is None:
            continue

        # update the database
        sql_string = f"UPDATE images_extracted SET complexity={complexity} WHERE image_id='{image_id}'"
        ctd.execute_sql(sql_string, conn)

        updated_entries += 1


if __name__ == "__main__":
    calc_complexities()
