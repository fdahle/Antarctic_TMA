# Library imports
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.estimate.estimate_height as eh


def estimate_heights():
    """"""

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images and heights from the database
    sql_string = "SELECT image_id, height, height_estimated FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # remove all entries that have a height
    data = data[data['height'].isnull()]

    # save how many entries are updated
    updated_entries = 0

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        # update the progress bar
        pbar.set_postfix_str(f"Estimate height for {image_id} "
                             f"({updated_entries} already updated)")

        # estimate the height
        estimated_height = eh.estimate_height(image_id, conn)

        # skip if the height could not be estimated
        if estimated_height is None:
            continue

        # update the database
        sql_string = f"UPDATE images_extracted SET " \
                     f"height={estimated_height}, " \
                     f"height_estimated=TRUE " \
                     f"WHERE image_id='{image_id}'"
        ctd.execute_sql(sql_string, conn)

        # increase the counter
        updated_entries += 1


if __name__ == "__main__":
    estimate_heights()
