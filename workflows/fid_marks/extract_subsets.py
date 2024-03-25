import src.base.connect_to_database as ctd
import src.fid_marks.extract_subset as es

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

    # loop over all images
    for idx, row in data.iterrows():

        # get the image id
        image_id = row['image_id']

        # loop over all subsets
        for subset in ['n', 'e', 's', 'w']:

            # get the subset values
            subset_x = row[f'subset_{subset}_x']
            subset_y = row[f'subset_{subset}_y']
            subset_estimated = row[f'subset_{subset}_estimated']
            subset_extraction_date = row[f'subset_{subset}_extraction_date']

            # if subset is not estimated and has an extraction date
            if not subset_estimated and subset_extraction_date is not None:

                # update the subset values
                sql_string = f"UPDATE images_fid_points " \
                             f"SET subset_{subset}_x = NULL, " \
                             f"subset_{subset}_y = NULL, " \
                             f"subset_{subset}_estimated = True " \
                             f"WHERE image_id = '{image_id}'"
                ctd.execute_sql(sql_string, conn)


if __name__ == "__main__":
    extract_subsets()
