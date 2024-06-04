# Package imports
from tqdm import tqdm

# Custom imports
import src.base.connect_to_database as ctd
import src.estimate.estimate_cam_id as eci

# Variables
overwrite = False


def estimate_cam_ids():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images and cam ids from the database
    sql_string = "SELECT image_id, cam_id, cam_id_estimated FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # remove all entries that have a cam id
    data = data[data['cam_id'].isnull()]

    # save how many entries are updated
    updated_entries = 0

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Estimate cam id for {image_id} "
                             f"({updated_entries} already updated)")

        # estimate the cam id
        estimated_cam_id = eci.estimate_cam_id(image_id, conn)

        if estimated_cam_id is None:
            continue

        # update the database
        sql_string = f"UPDATE images_extracted SET " \
                     f"cam_id='{estimated_cam_id}', " \
                     f"cam_id_estimated=TRUE " \
                     f"WHERE image_id='{image_id}'"

        ctd.execute_sql(sql_string, conn)

        updated_entries += 1


if __name__ == "__main__":
    estimate_cam_ids()
