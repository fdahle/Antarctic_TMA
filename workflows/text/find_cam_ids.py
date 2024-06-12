# Library imports
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.text.find_cam_id as fci

# variables
overwrite = False


def find_cam_ids():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images with their text and cam id from the database
    sql_string = "SELECT image_id, cam_id, cam_id_estimated, " \
                 "text_content FROM images_extracted WHERE text_content IS NOT NULL"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # filter out the images that already have a cam id
    if overwrite is False:
        data = data[data['cam_id'].isnull() | data['cam_id_estimated']]

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Find cam id for {image_id} "
                             f"({updated_entries} already updated)")

        # get the text
        text = row['text_content']

        # extract the cam id from the text
        cam_id = fci.find_cam_id(text)

        if cam_id is None:
            continue

        # update the database
        sql_string = f"UPDATE images_extracted SET " \
                     f"cam_id='{cam_id}', cam_id_estimated=FALSE" \
                     f" WHERE image_id='{image_id}'"

        ctd.execute_sql(sql_string, conn)

        updated_entries += 1

if __name__ == "__main__":
    find_cam_ids()