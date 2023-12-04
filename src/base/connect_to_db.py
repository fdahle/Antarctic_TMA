import json
import os.path
import pandas as pd
import psycopg2
import pyodbc
import sqlite3
import warnings

import base.print_v as p


def __build_connection(db_type, path_db_file=None):

    """
    __build_connection(db_type):
    This function builds up a connection to a specified database and is usually called before each operation.
    Note that this function should only be called from inside this file.
    Args:
        db_type (String): Specifies which database should be called (must be one of ["PSQL", "ACCESS", "SQLITE"]
        path_db_file (String, None): contains the path to the db file. If this is none, a default value is used.
    Returns:
        conn (Conn-Class): The Connection to the database
        cursor(Cursor-Class): The Cursor required to access a db
    """

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # connection to different databases must be build up different
    if db_type == "PSQL":
        conn = psycopg2.connect(database=json_data["psql_database"],
                                user=json_data["psql_user"],
                                password=json_data["psql_password"],
                                host=json_data["psql_host"],
                                port=json_data["psql_port"])
        cursor = conn.cursor()

    elif db_type == "ACCESS":

        # get path of the access database
        if path_db_file is None:
            default_access_path = None
        else:
            default_access_path = path_db_file

        assert os.path.exists(default_access_path), f"No file could be found at '{default_access_path}'"
        assert default_access_path.split(".")[-1] == "accdb", \
            f"The file at '{default_access_path}' must be an .accdb file"

        conn = pyodbc.connect(
            r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};' +
            'DBQ=' + default_access_path + ';')
        cursor = conn.cursor()

    elif db_type == "SQLITE":

        # get path of sql-lite db
        # get path of the access database
        if path_db_file is None:
            default_sqlite_path = json_data["path_file_sql_lite"]
        else:
            default_sqlite_path = path_db_file

        assert os.path.exists(default_sqlite_path), f"No file could be found at '{default_sqlite_path}'"
        assert default_sqlite_path.split(".")[-1] == "sqlite", \
            f"The file at '{default_sqlite_path}' must be an .sqlite file"

        # download data into an offline version
        create_sqllite_file()

        conn = sqlite3.connect(default_sqlite_path)
        cursor = conn.cursor()

    else:
        conn = None
        cursor = None

    return conn, cursor


# change the flag so that the local database is download again
def __change_flag():

    """
    __change_flag()
    This function is called whenever something is added/updated in the db. This means a flag file is changed to True.
    Whenever this flag-file is True, the sql-lite file is updated before it is used.
    Args:

    Returns:

    """

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the path to the flag
    default_flag_path = json_data["path_file_flag"]

    # change the flag
    with open(default_flag_path) as f:
        # get the file data
        file_data = f.read()

    # replace the flag
    file_data = file_data.replace('False', 'True')

    # rewrite back to file
    with open(default_flag_path, 'w') as f:
        f.write(file_data)


# extract data from db
def get_data_from_db(sql_string, db_type="PSQL", catch=True, verbose=False, pbar=None):

    """
    get_data_from_db(sql_string, db_type, verbose, catch):
    This function allows to get data from a database.
    Args:
        sql_string (String): A valid sql-string that describes which data is required.
        db_type (String, "PSQL"): Specifies which database should be called (must be one of ["PSQL", "ACCESS", "SQLITE"]
        catch (Boolean, True): If true, invalid sql-strings will just be ignored, otherwise a crash happens
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar

    Returns:
        data_frame (Pandas dataframe): The data from the database as a pandas dataframe. If something went wrong and
        catch=True, 'None' will be returned
    """

    assert db_type in ["PSQL", "ACCESS", "FILES"], "The specified database is incorrect"

    p.print_v(f"Execute {sql_string} at {db_type}", verbose, pbar=pbar)

    # get the connection
    conn, cursor = __build_connection(db_type)

    try:

        # ignore the pandas sqlAlchemy warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_frame = pd.read_sql(sql_string, conn)
        return data_frame
    except (Exception,) as e:
        if catch:
            return None
        else:
            raise e


# add data to db
def add_data_to_db(sql_string, db_type="PSQL", add_timestamp=True,
                   catch=True, verbose=False, pbar=None):

    """
    add_data_to_db:
    This function allows to add data to a database.
    Args:
        sql_string (String): A valid sql-string that describes which data will be inserted.
        db_type (String, "PSQL"): Specifies which database should be called (must be one of ["PSQL", "ACCESS", "SQLITE"]
        add_timestamp(Boolean, True): If true, the "last_change" field will be updated in a table
        catch (Boolean, True): If true, invalid sql-strings will just be ignored, otherwise a crash happens
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        success (Boolean): Returns 'True' for a successful adding
    """

    assert db_type in ["PSQL", "ACCESS", "FILES"], "The specified database is incorrect"

    p.print_v(f"Execute {sql_string} at {db_type}", verbose, pbar=pbar)

    # edit and add is the same, but so that people have the possibility to call what they want
    bool_success = edit_data_in_db(sql_string, db_type, add_timestamp, catch, verbose=False)

    return bool_success


# change data in db
def edit_data_in_db(sql_string, db_type="PSQL", add_timestamp=True,
                    catch=True, verbose=False, pbar=None):

    """
    edit_data_in_db:
    This function allows to edit data in a database.
    Args:
        sql_string (String): A valid sql-string that describes which data will be edited.
        db_type (String, "PSQL"): Specifies which database should be called (must be one of ["PSQL", "ACCESS", "SQLITE"]
        add_timestamp(Boolean, True): If true, the "last_change" field will be updated in a table
        catch (Boolean, True): If true, invalid sql-strings will just be ignored, otherwise a crash happens
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        success (Boolean): Returns 'True' for a successful editing
    """

    if "INSERT INTO" not in sql_string and "DELETE" not in sql_string and "WHERE" not in sql_string:
        raise ValueError("WHERE IS MISSING FOR EDIT!")

    assert db_type in ["PSQL", "ACCESS", "FILES"], "The specified database is incorrect"

    p.print_v(f"Execute {sql_string} at {db_type}", verbose, pbar=pbar)

    # only add last change is not already in sql string, and we are not deleting
    if add_timestamp and "last_change" not in sql_string:

        if "INSERT INTO" in sql_string:
            # split the sql string so that we can add last data
            splits = sql_string.split(")")

            splits[0] = splits[0] + ", last_change)"
            splits[1] = splits[1] + ", NOW())"

            sql_string = splits[0] + splits[1]

        elif "UPDATE" in sql_string:

            splits = sql_string.split(" WHERE")

            splits[0] = splits[0] + ", last_change=NOW()"

            sql_string = splits[0] + " WHERE" + splits[1]

    # get the connection
    conn, cursor = __build_connection(db_type)

    try:
        # execute the edit
        cursor.execute(sql_string)
        conn.commit()

        # change the flag to update that the db has changed
        __change_flag()
        return True

    except (Exception,) as e:
        if catch:
            return False
        else:
            raise e


# delete data from db
def delete_data_from_db(sql_string, db_type="PSQL",
                        catch=True, verbose=False, pbar=None):

    """
    delete_data_from_db:
    This function allows to delete data from a database.
    Args:
        sql_string (String): A valid sql-string that describes which data will be deleted.
        db_type (String, "PSQL"): Specifies which database should be called (must be one of ["PSQL", "ACCESS", "SQLITE"]
        catch (Boolean, True): If true, invalid sql-strings will just be ignored, otherwise a crash happens
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        success (Boolean): Returns 'True' for a successful deleting
    """

    assert db_type in ["PSQL", "ACCESS", "FILES"], "The specified database is incorrect"

    p.print_v(f"Execute {sql_string} at {db_type}", verbose, pbar=pbar)

    # edit and delete is the same, but so that people have the possibility to call what they want
    bool_success = edit_data_in_db(sql_string, db_type, catch, verbose=False)

    return bool_success


def create_sqllite_file(sql_lite_path=None, flag_path=None,
                        verbose=False, pbar=None):

    """
    create_sqllite_file(sql_lite_path, db_name, flag_path, verbose):
    This function is used to keep the sql-lite file up-to-date. In the beginning it checks the specified
    flag file (under flag_path). The flag-file is just a txt-file with either 'True' or 'False'. False means
    that the sql-lite must be updated. (The change in the flag file happens in 'connect_to_db.py').
    If the sql-lite must be updated, all data from PSQL is downloaded and stored in the sql-lite file.

    Args:
        sql_lite_path (String, None): The path where the sql-lite file is. If none, the default path is taken.
        flag_path (String, None): The path where the flag-file is located. If None, the default path is taken.
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        None
    """

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # init default values
    default_flag_path = json_data["path_file_flag"]
    default_sql_lite_path = json_data["path_file_sql_lite"]

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
        p.print_v("Database is up to date and no download is required!", verbose, pbar=pbar)
        return

    p.print_v("Database must be updated!", verbose, pbar=pbar)

    # set default values
    if sql_lite_path is None:
        sql_lite_path = default_sql_lite_path

    # create db
    conn = sqlite3.connect(sql_lite_path)

    # create sql string to get all tables of database
    sql_string = "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"

    p.print_v("Download all table names", verbose, pbar=pbar)

    # download these tables
    tables = get_data_from_db(sql_string)

    # iterate all tables
    for elem in tables["table_name"]:

        p.print_v(f"Download all data from {elem}", verbose, pbar=pbar)

        # create sql string and download data
        sql_string = "SELECT * FROM " + elem
        table_data = get_data_from_db(sql_string)

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

    p.print_v("Finished creating sql_lite file", verbose=verbose, pbar=pbar)
