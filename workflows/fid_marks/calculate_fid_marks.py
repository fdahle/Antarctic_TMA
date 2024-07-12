# Library imports
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.display.display_images as di
import src.fid_marks.calculate_fid_mark as cfm
import src.load.load_image as li

# Variables
overwrite = False

# Debug variables
debug_display_fid_marks = True


def calculate_fid_marks():
    # Position of fid marks:
    # 3 7 2
    # 5   6
    # 1 8 4
    #

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images and fid-marks from the database
    sql_string = "SELECT * FROM images_fid_points"
    fid_data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    #fid_data = fid_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # loop over all images
    for idx, row in (pbar := tqdm(fid_data.iterrows(), total=fid_data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Calculate fiducial marks for {image_id} "
                             f"({updated_entries} already updated)")

        # reset the image variable
        image = None

        subset_width = row["subset_width"]
        subset_height = row["subset_height"]

        fid_marks = []

        # iterate over all keys
        for key in ["ne", "nw", "se", "sw"]:

            pbar.set_postfix_str(f"Extract fid mark {key} for {image_id} "
                                 f"({updated_entries} already updated)")

            # convert key to numeric key for fid marks
            numeric_key = {'ne': 2, 'nw': 3, 'se': 4, 'sw': 1}[key]

            # get existing fid mark information
            fid_mark_estimated = row[f'fid_mark_{numeric_key}_estimated']
            fid_mark_extraction_date = row[f'fid_mark_{numeric_key}_extraction_date']

            # if fid mark is not estimated and has an extraction date
            if fid_mark_estimated or fid_mark_extraction_date is None or overwrite:
                pass
            else:
                continue

            # split the key in the two directions
            direction1 = list(key)[0]
            direction2 = list(key)[1]

            # get the subset bounds for both directions
            subset1_x = row[f"subset_{direction1}_x"]
            subset1_y = row[f"subset_{direction1}_y"]
            subset2_x = row[f"subset_{direction2}_x"]
            subset2_y = row[f"subset_{direction2}_y"]

            # skip if subset is not existing
            if pd.isnull(subset1_x) or pd.isnull(subset1_y) or \
                    pd.isnull(subset2_x) or pd.isnull(subset2_y) or \
                    pd.isnull(subset_width) or pd.isnull(subset_height):
                continue

            # create bounds for the subsets
            subset1 = (subset1_x, subset1_y, subset1_x + subset_width, subset1_y + subset_height)
            subset2 = (subset2_x, subset2_y, subset2_x + subset_width, subset2_y + subset_height)

            # create a list of subsets
            subsets = [subset1, subset2]

            # load the image
            if image is None:
                try:
                    image = li.load_image(image_id)
                except (Exception,):
                    continue

            # calculate the fid mark
            try:
                fid_mark = cfm.calculate_fid_mark(image, key, subsets)
            except (Exception,):
                print("An error happened at", image_id, key)
                continue

            # skip if no fid mark is found
            if fid_mark is None:
                continue

            fid_marks.append(fid_mark)

            # Get the current date
            current_date = datetime.now()

            # update the subset values
            sql_string = f"UPDATE images_fid_points " \
                         f"SET fid_mark_{numeric_key}_x={fid_mark[0]}, " \
                         f"fid_mark_{numeric_key}_y={fid_mark[1]}, " \
                         f"fid_mark_{numeric_key}_extraction_date='{current_date}', " \
                         f"fid_mark_{numeric_key}_estimated=False " \
                         f"WHERE image_id='{image_id}'"

            ctd.execute_sql(sql_string, conn)
            updated_entries += 1

        if debug_display_fid_marks:
            di.display_images([image], points=[fid_marks])


if __name__ == "__main__":
    calculate_fid_marks()
