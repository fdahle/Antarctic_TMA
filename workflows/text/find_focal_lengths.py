# Library imports
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.text.find_focal_length as ffl

# variables
overwrite = True


def find_focal_lengths():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images with their text and focal length from the database
    sql_string = "SELECT image_id, focal_length, focal_length_estimated, " \
                 "text_content FROM images_extracted WHERE text_content IS NOT NULL"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # filter out the images that already have a focal length
    if overwrite is False:
        data = data[data['focal_length'].isnull() | data['focal_length_estimated']]

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Find focal length for {image_id} "
                             f"({updated_entries} already updated)")

        # get the text
        text = row['text_content']

        # extract the focal length from the text
        # focal_length = ffl.find_focal_length(text)
        print(text)
        focal_length = ffl.find_focal_length(text, method="text")

        if focal_length is None:
            continue

        # update the database
        #sql_string = f"UPDATE images_extracted SET " \
        #             f"focal_length='{focal_length}', focal_length_estimated=FALSE" \
        #             f" WHERE image_id='{image_id}'"
        sql_string = f"UPDATE images_extracted SET " \
                     f"focal_length_text='{focal_length}'" \
                     f" WHERE image_id='{image_id}'"

        ctd.execute_sql(sql_string, conn)

        updated_entries += 1


if __name__ == "__main__":
    find_focal_lengths()
