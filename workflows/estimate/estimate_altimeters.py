# Library imports
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.estimate.estimate_altimeter as ea
import src.load.load_image as li

# Variables
overwrite = False


def estimate_altimeters():
    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images and altimeters from the database
    sql_string = "SELECT image_id, " \
                 "altimeter_x, altimeter_y, " \
                 "altimeter_width, altimeter_height, " \
                 "altimeter_value " \
                 "FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        # get the image
        try:
            image = li.load_image(image_id)
        except Exception as e:
            continue

        pbar.set_postfix_str(f"Estimate altimeter for {image_id} "
                             f"({updated_entries} already updated)")

        # estimate the altimeter values
        try:
            estimated_altimeter, altimeter_position = ea.extract_altimeter(image, return_position=True)
        except Exception as e:
            continue

        # check if the altimeter position was found
        if altimeter_position is None:
            continue

        # check if we need to save the altimeter position
        if overwrite is True or row['altimeter_x'] is None or row['altimeter_y'] is None or \
                row['altimeter_width'] is None or row['altimeter_height'] is None:

            # create sql_string
            sql_string = f"UPDATE images_extracted SET " \
                         f"altimeter_x={altimeter_position[0]}, " \
                         f"altimeter_y={altimeter_position[1]}, " \
                         f"altimeter_width={altimeter_position[2]}, " \
                         f"altimeter_height={altimeter_position[3]} " \
                         f"WHERE image_id='{image_id}'"
            ctd.execute_sql(sql_string, conn)

        # check if an altimeter value was found
        if estimated_altimeter is None:
            continue

        if overwrite is True or row['altimeter_value'] is None:

            # create sql_string
            sql_string = f"UPDATE images_extracted SET " \
                         f"altimeter_value={estimated_altimeter} " \
                         f"WHERE image_id='{image_id}'"
            ctd.execute_sql(sql_string, conn)

        updated_entries += 1

if __name__ == "__main__":
    estimate_altimeters()
