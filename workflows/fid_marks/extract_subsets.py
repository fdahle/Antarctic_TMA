# Library imports
from datetime import datetime
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.load.load_image as li
import src.fid_marks.extract_subset as es

# Variables
overwrite = False
binarize=False
refine_multiple=True

def extract_subsets():
    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images and subsets from the database
    sql_string = "SELECT image_id, " \
                 "subset_width, subset_height, " \
                 "subset_n_x, subset_n_y, subset_n_estimated, subset_n_extraction_date, " \
                 "subset_e_x, subset_e_y, subset_e_estimated, subset_e_extraction_date, " \
                 "subset_s_x, subset_s_y, subset_s_estimated, subset_s_extraction_date, " \
                 "subset_w_x, subset_w_y, subset_w_estimated, subset_w_extraction_date " \
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

            pbar.set_postfix_str(f"Extract subset {key} for {image_id} "
                                 f"({updated_entries} already updated)")

            # get existing subset information
            subset_estimated = row[f'subset_{key}_estimated']
            subset_extraction_date = row[f'subset_{key}_extraction_date']

            # if subset is not estimated and has an extraction date
            if overwrite or subset_estimated or subset_extraction_date is None:

                # load the image
                if image is None:
                    try:
                        image = li.load_image(image_id)
                    except (Exception,):
                        continue

                # extract the subset
                bbox = es.extract_subset(image, key,
                                         binarize_crop=binarize, refine_multiple=refine_multiple)

                # skip if no bbox is found
                if bbox is None:
                    continue

                # Get the current date
                current_date = datetime.now()

                # update the subset values
                sql_string = f"UPDATE images_fid_points " \
                             f"SET subset_{key}_x={bbox[0]}, " \
                             f"subset_{key}_y={bbox[2]}, " \
                             f"subset_width=250, subset_height=250, " \
                             f"subset_{key}_extraction_date='{current_date}', " \
                             f"subset_{key}_estimated=False " \
                             f"WHERE image_id='{image_id}'"
                ctd.execute_sql(sql_string, conn)
                updated_entries += 1


if __name__ == "__main__":
    extract_subsets()
