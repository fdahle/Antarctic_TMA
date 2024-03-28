import pandas as pd

from datetime import datetime
from tqdm import tqdm

import src.base.connect_to_database as ctd
import src.load.load_image as li
import src.fid_marks.extract_fid_mark as efm

overwrite = False


def extract_fid_marks():
    # Position of fid marks:
    # 3 7 2
    # 5   6
    # 1 8 4
    #

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images, subsets and fid-marks from the database
    sql_string = "SELECT image_id, " \
                 "subset_width, subset_height, " \
                 "subset_n_x, subset_n_y, subset_n_estimated, subset_n_extraction_date, " \
                 "subset_e_x, subset_e_y, subset_e_estimated, subset_e_extraction_date, " \
                 "subset_s_x, subset_s_y, subset_s_estimated, subset_s_extraction_date, " \
                 "subset_w_x, subset_w_y, subset_w_estimated, subset_w_extraction_date " \
                 "fid_mark_5_x, fid_mark_5_y, fid_mark_5_estimated, fid_mark_5_extraction_date, " \
                 "fid_mark_6_x, fid_mark_6_y, fid_mark_6_estimated, fid_mark_6_extraction_date, " \
                 "fid_mark_7_x, fid_mark_7_y, fid_mark_7_estimated, fid_mark_7_extraction_date, " \
                 "fid_mark_8_x, fid_mark_8_y, fid_mark_8_estimated, fid_mark_8_extraction_date " \
                 "FROM images_fid_points"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        # reset the image variable
        image = None

        # loop over all subsets
        for key in ['n', 'e', 's', 'w']:

            # get the subset values
            subset_x = row[f'subset_{key}_x']
            subset_y = row[f'subset_{key}_y']
            subset_width = row['subset_width']
            subset_height = row['subset_height']

            # skip if subset is not existing
            if pd.isnull(subset_x) or pd.isnull(subset_y) or \
                    pd.isnull(subset_width) or pd.isnull(subset_height):
                continue

            # convert key to numeric key for fid marks
            numeric_key = {'n': 7, 'e': 6, 's': 8, 'w': 5}[key]

            pbar.set_postfix_str(f"Extract fid mark {key} for {image_id} "
                                 f"({updated_entries} already updated)")

            # get existing fid mark information
            fid_mark_estimated = row[f'fid_mark_{numeric_key}_estimated']
            fid_mark_extraction_date = row[f'fid_mark_{numeric_key}_extraction_date']

            # if fid mark is not estimated and has an extraction date
            if overwrite or fid_mark_estimated or fid_mark_extraction_date is None:

                # load the image
                if image is None:
                    try:
                        image = li.load_image(image_id)
                    except (Exception,):
                        continue

                # get bounding box of the subset
                subset_bounds = (int(subset_x), int(subset_y),
                                int(subset_x + subset_width), int(subset_y + subset_height))

                # extract the fid mark
                try:
                    point = efm.extract_fid_mark(image, key, subset_bounds)
                except (Exception,):
                    point = None

                # skip if no fid mark is found
                if point is None:
                    continue

                # Get the current date
                current_date = datetime.now()

                # update the subset values
                sql_string = f"UPDATE images_fid_points " \
                             f"SET fid_mark_{numeric_key}_x={point[0]}, " \
                             f"fid_mark_{numeric_key}_y={point[1]}, " \
                             f"fid_mark_{numeric_key}_extraction_date='{current_date}', " \
                             f"fid_mark_{numeric_key}_estimated=False " \
                             f"WHERE image_id='{image_id}'"
                ctd.execute_sql(sql_string, conn)
                updated_entries += 1

if __name__ == "__main__":
    extract_fid_marks()
