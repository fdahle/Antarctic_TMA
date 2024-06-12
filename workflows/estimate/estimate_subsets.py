# Library imports
from datetime import datetime
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.estimate.estimate_subset as es

# Variables
overwrite = False


def estimate_subsets():
    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images and subsets from the database
    sql_string = "SELECT image_id, " \
                 "subset_width, subset_height, " \
                 "subset_n_x, subset_n_y, subset_n_estimated, " \
                 "subset_e_x, subset_e_y, subset_e_estimated, " \
                 "subset_s_x, subset_s_y, subset_s_estimated, " \
                 "subset_w_x, subset_w_y, subset_w_estimated " \
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

        pbar.set_postfix_str(f"Estimate subsets for {image_id} "
                             f"({updated_entries} already updated)")

        # init variables for checking the subsets
        estimated_subsets = {}  # estimated subset positions
        subset_data = None  # subset data for that image

        # check all subsets
        for key in ['n', 'e', 's', 'w']:

            # check if the subset was found the original way
            if row[f'subset_{key}_estimated'] is False:
                estimated_subsets[key] = None
                continue

            # check if we should overwrite the subset
            if overwrite is False and row[f'subset_{key}_x'] is True:
                estimated_subsets[key] = None
                continue

            # estimate the subset
            subset_coords, subset_data = es.estimate_subset(image_id, key,
                                                            return_data=True,
                                                            subset_data=subset_data, conn=conn)
            estimated_subsets[key] = subset_coords

        # skip updating if all subsets are None
        if all([val is None for val in estimated_subsets.values()]):
            continue

        # Get the current date
        current_date = datetime.now()

        # create update_string
        sql_string = "UPDATE images_fid_points SET "
        for key in ['n', 'e', 's', 'w']:
            if estimated_subsets[key] is not None:
                sql_string += f"subset_{key}_x={estimated_subsets[key][0]}, " \
                              f"subset_{key}_y={estimated_subsets[key][1]}, " \
                              f"subset_{key}_estimated=True, " \
                              f"subset_{key}_extraction_date='{current_date}', "
        sql_string = sql_string[:-2]
        sql_string += f" WHERE image_id='{image_id}'"

        # update the database
        ctd.execute_sql(sql_string, conn)

        updated_entries += 1


if __name__ == "__main__":
    estimate_subsets()
