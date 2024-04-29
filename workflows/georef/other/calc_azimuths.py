# Package imports
from tqdm import tqdm

# Custom imports
import src.base.connect_to_database as ctd
import src.georef.snippets.calc_azimuth as ca

# Variables
overwrite = False


def calc_azimuths():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all image ids
    sql_string = "SELECT image_id, azimuth_exact FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        # check if azimuth is already calculated
        if row['azimuth_exact'] is not None and overwrite is False:
            continue

        # set description for tqdm
        pbar.set_postfix_str(f"Calculate azimuth for {image_id} "
                             f"({updated_entries} already updated)")

        # calculate the azimuth
        azimuth = ca.calc_azimuth(image_id, conn)

        # skip if the azimuth is None
        if azimuth is None:
            continue

        # create update string
        sql_string = f"UPDATE images_extracted SET azimuth_exact = {azimuth} " \
                     f"WHERE image_id = '{image_id}'"

        # update the database
        ctd.execute_sql(sql_string, conn)

        # update the counter
        updated_entries += 1


if __name__ == "__main__":
    calc_azimuths()
