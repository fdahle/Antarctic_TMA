import json
import os
import pandas as pd

from tqdm import tqdm

import base.connect_to_db as ctd
import base.print_v as p

debug_more_details = True
debug_max_cutoff = 100

_tables = ["images_extracted"]


def quality_check_tables(tables=None):
    """
    quality_check_tables(tables):
    This function performs a basic quality check for tables by checking the following:
     - are there any double entries in the tables
     - are there entries missing in the tables
    Args:
        tables:

    Returns:

    """


    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if tables is None:
        tables = json_data["check_tables_tables"]

    for table in tables:

        p.print_v(f"Check table '{table}'")

        # check first for double entries
        sql_string = f"SELECT image_id FROM {table} GROUP BY image_id HAVING COUNT(image_id) > 1"
        data = ctd.get_data_from_db(sql_string)
        if data.shape[0] > 0:
            p.print_v(f"There are {data.shape[0]} double entries in table '{table}'", color="red")
            if debug_more_details:
                data_lst = data['image_id'].tolist()
                if len(data_lst) > debug_max_cutoff:
                    p.print_v(data_lst[0:debug_max_cutoff])
                    p.print_v(f"list was cut off after {debug_max_cutoff} entries")
                else:
                    p.print_v(data_lst)

        else:
            p.print_v(f"There are no double entries in table '{table}'", color="green")

        # get data
        sql_string = f"SELECT * FROM {table}"
        data = ctd.get_data_from_db(sql_string)

        # get total number of entries
        total_number_of_entries = data.shape[0]

        # create dict of lists for every column
        col_dict = {}
        for col in data.columns.values.tolist():
            col_dict[col] = []

        # remove some unnecessary entries from dict
        try:
            del col_dict["image_id"]
            del col_dict["comment"]
            del col_dict["last_change"]
        except (Exception,):
            pass

        # iterate all rows
        for index in (pbar := tqdm(range(data.shape[0]))):

            row = data.iloc[index]

            for col in row.keys():

                # exclude some keys
                if col in ["image_id", "comment", "last_change"]:
                    continue

                # get value
                val = row[col]

                # check if value is empty
                if val is None or pd.isnull(val):
                    col_dict[col].append(row["image_id"])

            p.print_v(f"Check all entries for table '{table}'", pbar=pbar)

        for key in col_dict.keys():
            percentage = round(len(col_dict[key]) / total_number_of_entries * 100, 2)

            if len(col_dict[key]) > 0:
                p.print_v(f" - {key}: {len(col_dict[key])} entries missing ({percentage}%)", color="red")
                if debug_more_details:
                    if len(col_dict[key]) > debug_max_cutoff:
                        p.print_v(col_dict[key][0:debug_max_cutoff])
                        p.print_v(f"list was cut off after {debug_max_cutoff} entries")
                    else:
                        p.print_v(col_dict[key])

            else:
                p.print_v(f" - {key}: No entries are missing", color="green")

        print("")

if __name__ == "__main__":

    quality_check_tables(_tables)