# Library imports
from datetime import datetime
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.estimate.estimate_fid_mark as efm

# Variables
overwrite = False


def estimate_fid_marks():
    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images and fid-marks from the database
    sql_string = "SELECT image_id, " \
                 "fid_mark_1_x, fid_mark_1_y, fid_mark_1_estimated, " \
                 "fid_mark_2_x, fid_mark_2_y, fid_mark_2_estimated, " \
                 "fid_mark_3_x, fid_mark_3_y, fid_mark_3_estimated, " \
                 "fid_mark_4_x, fid_mark_4_y, fid_mark_4_estimated, " \
                 "fid_mark_5_x, fid_mark_5_y, fid_mark_5_estimated, " \
                 "fid_mark_6_x, fid_mark_6_y, fid_mark_6_estimated, " \
                 "fid_mark_7_x, fid_mark_7_y, fid_mark_7_estimated, " \
                 "fid_mark_8_x, fid_mark_8_y, fid_mark_8_estimated " \
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

        pbar.set_postfix_str(f"Estimate fid marks for {image_id} "
                             f"({updated_entries} already updated)")

        # init variables for checking the fid marks
        estimated_fid_marks = {}
        fid_mark_data = None

        # check all fid marks
        for key in [1, 2, 3, 4, 5, 6, 7, 8]:

            key = str(key)

            # check if the fid mark was found the original way
            if row[f'fid_mark_{key}_estimated'] is False:
                estimated_fid_marks[key] = None
                continue

            # check if we should overwrite the fid mark
            if overwrite is False and row[f'fid_mark_{key}_x'] is True:
                estimated_fid_marks[key] = None
                continue

            # estimate the fid mark
            fid_mark_coords, fid_mark_data = efm.estimate_fid_mark(image_id, key,
                                                                   return_data=True,
                                                                   fid_mark_data=fid_mark_data,
                                                                   conn=conn)
            estimated_fid_marks[key] = fid_mark_coords

        # skip updating if all fid marks are None
        if all([val is None for val in estimated_fid_marks.values()]):
            continue

        # get the current date
        current_date = datetime.now()

        # create update_string
        sql_string = "UPDATE images_fid_points SET "
        for key in [1, 2, 3, 4, 5, 6, 7, 8]:
            key = str(key)
            if estimated_fid_marks[key] is not None:
                sql_string += f"fid_mark_{key}_x={estimated_fid_marks[key][0]}, " \
                              f"fid_mark_{key}_y={estimated_fid_marks[key][1]}, " \
                              f"fid_mark_{key}_estimated=True, " \
                              f"fid_mark_{key}_extraction_date='{current_date}', "
        sql_string = sql_string[:-2]
        sql_string += f" WHERE image_id={image_id}"

        # update the database
        ctd.execute_sql(sql_string, conn)

        updated_entries += 1


if __name__ == "__main__":
    estimate_fid_marks()
