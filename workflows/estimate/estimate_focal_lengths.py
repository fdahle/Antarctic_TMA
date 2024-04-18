# Package imports
from tqdm import tqdm

# Custom imports
import src.base.connect_to_database as ctd
import src.estimate.estimate_focal_length as efl


def estimate_focal_lengths():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images and focal lengths from the database
    sql_string = "SELECT image_id, focal_length, focal_length_estimated FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # remove all entries that have a focal length
    data = data[data['focal_length'].isnull()]

    # save how many entries are updated
    updated_entries = 0

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Estimate focal length for {image_id} "
                             f"({updated_entries} already updated)")

                # estimate the focal length
        estimated_focal_length = efl.estimate_focal_length(image_id, conn)

        if estimated_focal_length is None:
            continue

        # update the database
        sql_string = f"UPDATE images_extracted SET " \
                     f"focal_length={estimated_focal_length}, " \
                     f"focal_length_estimated=TRUE " \
                     f"WHERE image_id='{image_id}'"

        ctd.execute_sql(sql_string, conn)

        updated_entries += 1


if __name__ == "__main__":
    estimate_focal_lengths()
