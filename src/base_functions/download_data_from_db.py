import sqlite3

import connect_to_db as cdb

"""
download_data_from_db(sql_lite_path, db_name, flag_path, verbose):
This function is used to keep the sql-lite file up-to-date. In the begin it checks the specified
flag file (under flag_path). The flag-file is just a txt-file with either 'True' or 'False'. False means
that the sql-lite must be updated. (The change in the flag file happens in 'connect_to_db.py').
If the sql-lite must be updated, all data from PSQL is downloaded and stored in the sql-lite file.

INPUT:
    sql_lite_path (String, None): The path where the sql-lite file is. If none, the default path is taken.
    db_name (String, None): The name of the sql_lite file. If None, the default name is taken.
    flag_path (String, None): The path where the flag-file is located. If None, the default path is taken.
OUTPUT:
"""


def download_data_from_db(sql_lite_path=None, db_name=None, flag_path=None, verbose=False):

    # init values
    default_flag_path = "/media/fdahle/beb5a64a-5335-424a-8f3c-779527060523/ATM/data/databases/sqlite/flag.txt"
    default_sql_lite_path = "/media/fdahle/beb5a64a-5335-424a-8f3c-779527060523/ATM/data/databases/sqlite"
    default_db_name = "sqlite"

    # set default value
    if flag_path is None:
        flag_path = default_flag_path

    # check if it is necessary to update the database
    with open(flag_path) as f:
        lines = f.readlines()

        # flag is in last line
        flag_line = lines[-1]
        flag = flag_line.split("=")[1]

    # if no change has happened nothing needs to be done
    if flag == "False":
        if verbose:
            print("Database is up to date and no download is required!")
        return

    if verbose:
        print("Database must be updated!")

    # set default values
    if sql_lite_path is None:
        sql_lite_path = default_sql_lite_path
    if db_name is None:
        db_name = default_db_name

    # create db
    conn = sqlite3.connect(sql_lite_path + "/" + db_name + ".db")

    # create sql string to get all tables of database
    sql_string = "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"

    if verbose:
        print("Download all table names")

    # download these tables
    tables = cdb.get_data_from_db(sql_string)

    # iterate all tables
    for elem in tables["table_name"]:

        if verbose:
            print(f"Download all data from {elem}")

        # create sql string and download data
        sql_string = "SELECT * FROM " + elem
        table_data = cdb.get_data_from_db(sql_string)

        # push data to sqlite
        table_data.to_sql(elem, conn, if_exists="replace")

    # change the flag
    with open(default_flag_path) as f:
        # get the file data
        file_data = f.read()

    # replace the flag
    file_data = file_data.replace('True', 'False')

    # rewrite back to file
    with open(default_flag_path, 'w') as f:
        f.write(file_data)


if __name__ == "__main__":
    download_data_from_db(verbose=True)
