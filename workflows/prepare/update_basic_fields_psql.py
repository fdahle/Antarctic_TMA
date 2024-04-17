# Package imports
from tqdm import tqdm

# Custom imports
import src.base.connect_to_database as ctd
import src.load.load_image as li

# variables
overwrite = False

def update_basic_fields_psql():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images from the database
    sql_string = "SELECT image_id, image_width, image_height FROM images_fid_points"
    data = ctd.execute_sql(sql_string, conn)

    # remove all entries that have a width and height
    if overwrite is False:
        data = data[data['image_width'].isnull() | data['image_height'].isnull()]

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Update basic fields for {image_id} "
                                f"({updated_entries} already updated)")

        # load the image
        try:
            image = li.load_image(image_id)
        except (Exception,):
            continue

        # get the image width and height
        image_width = image.shape[1]
        image_height = image.shape[0]

        # create sql string
        sql_string = f"UPDATE images_fid_points SET image_width={image_width}, image_height={image_height} " \
                     f"WHERE image_id='{image_id}'"

        # update the database
        ctd.execute_sql(sql_string, conn)

        # update the counter
        updated_entries += 1


if __name__ == "__main__":
    update_basic_fields_psql()
