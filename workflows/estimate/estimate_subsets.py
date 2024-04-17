# Package imports
from tqdm import tqdm

# Custom imports
import src.base.connect_to_database as ctd

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




if __name__ == "__main__":

    estimate_subsets()
