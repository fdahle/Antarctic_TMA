# Package imports
from tqdm import tqdm

# Custom imports
import src.base.connect_to_database as ctd
import src.text.find_height as fh

overwrite = False


def find_height_in_text():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images with their text from the database
    sql_string = "SELECT image_id, height, height_estimated, " \
                 "text_content FROM images_extracted WHERE text_content IS NOT NULL"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # filter out the images that already have a height
    if overwrite is False:
        data = data[data['height'].isnull() | data['height_estimated']]

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Find height for {image_id} "
                             f"({updated_entries} already updated)")

        # get the text
        text = row['text_content']

        # extract the height from the text
        height = fh.find_height(text)

        if height is None:
            continue

        # update the database
        sql_string = f"UPDATE images_extracted SET " \
                     f"height='{height}', height_estimated=FALSE" \
                     f" WHERE image_id='{image_id}'"

        ctd.execute_sql(sql_string, conn)

        updated_entries += 1


if __name__ == "__main__":
    find_height_in_text()
