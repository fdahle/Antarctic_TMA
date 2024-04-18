# Package imports
from tqdm import tqdm

# Custom imports
import src.base.connect_to_database as ctd
import src.estimate.estimate_subset as es

overwrite = False

def estimate_subsets():

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

            pbar.set_postfix_str(f"Estimate subsets for {image_id} "
                                f"({updated_entries} already updated)")

            # estimate the subsets
            estimated_subsets = estimate_subsets(image_id, conn)

            if estimated_subsets is None:
                continue

            # update the database
            sql_string = f"UPDATE images_fid_points SET " \
                        f"subset_n_x={estimated_subsets['n']['x']}, subset_n_y={estimated_subsets['n']['y']}, " \
                        f"subset_e_x={estimated_subsets['e']['x']}, subset_e_y={estimated_subsets['e']['y']}, " \
                        f"subset_s_x={estimated_subsets['s']['x']}, subset_s_y={estimated_subsets['s']['y']}, " \
                        f"subset_w_x={estimated_subsets['w']['x']}, subset_w_y={estimated_subsets['w']['y']} " \
                        f"WHERE image_id='{image_id}'"

            ctd.execute_sql(sql_string, conn)

            updated_entries += 1



if __name__ == "__main__":

    estimate_subsets()
