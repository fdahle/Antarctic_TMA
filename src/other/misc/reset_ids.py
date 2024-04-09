# Custom imports
import src.base.connect_to_database as ctd

# Variables
table = "images_fid_points"
columns = ["subset_n_x", "subset_n_y", "subset_n_estimated", "subset_n_extraction_date",
           "subset_e_x", "subset_e_y", "subset_e_estimated", "subset_e_extraction_date",
           "subset_s_x", "subset_s_y", "subset_s_estimated", "subset_s_extraction_date",
           "subset_w_x", "subset_w_y", "subset_w_estimated", "subset_w_extraction_date",
           "fid_mark_1_x", "fid_mark_1_y", "fid_mark_1_estimated", "fid_mark_1_extraction_date",
           "fid_mark_2_x", "fid_mark_2_y", "fid_mark_2_estimated", "fid_mark_2_extraction_date",
           "fid_mark_3_x", "fid_mark_3_y", "fid_mark_3_estimated", "fid_mark_3_extraction_date",
           "fid_mark_4_x", "fid_mark_4_y", "fid_mark_4_estimated", "fid_mark_4_extraction_date",
           "fid_mark_5_x", "fid_mark_5_y", "fid_mark_5_estimated", "fid_mark_5_extraction_date",
           "fid_mark_6_x", "fid_mark_6_y", "fid_mark_6_estimated", "fid_mark_6_extraction_date",
           "fid_mark_7_x", "fid_mark_7_y", "fid_mark_7_estimated", "fid_mark_7_extraction_date",
           "fid_mark_8_x", "fid_mark_8_y", "fid_mark_8_estimated", "fid_mark_8_extraction_date"]


def reset_ids(ids):
    """
    This function resets specified columns in specified tables for the given ids.
    """

    answer = input(f"There are {len(ids)} ids in the list. "
                   f"Press 'y' if you really want to reset the ids: ")
    if answer != 'y':
        return

    # create columns string
    column_string = ', '.join([f"{column}=NULL" for column in columns])

    # create ids string
    # Ensure each ID is quoted
    ids_formatted = ', '.join([f"'{id}'" for id in ids])

    # create final sql string
    sql_string = f"UPDATE {table} SET {column_string} " \
                 f"WHERE image_id IN ({ids_formatted})"

    print(sql_string)

    # establish connection
    conn = ctd.establish_connection()

    ctd.execute_sql(sql_string, conn)


if __name__ == "__main__":
    # get the ids
    with open('/data_1/ATM/rotation_ids.txt', 'r') as file:
        ids = [line.strip() for line in file]

    reset_ids(ids)
