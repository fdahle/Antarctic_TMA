import pandas as pd
import sqlite3
import psycopg2

import download_data_from_db as ddfd

"""
connect_to_db:
This file contains multiple functions all used to connect to the places where the data is stored.
Currently, there are three possible databases: PostGreSql, MS Access or a SQLLite database.
Following functions can be found:
    - __build_connection(db_type):
    This function builds up a connection to a specified database and is usually called before each operation.
    Note that this function should only be called from inside this file.
    INPUT:
        db_type (String): Specifies which database should be called (must be one of ["PSQL", "ACCESS", "SQLITE"]
    OUTPUT:
        conn (Conn-Class): The Connection to the database
        cursor(Cursor-Class): The Cursor required to access a db

    - get_data_from_db(sql_string, db_type, verbose, catch):
    This function allows to access data from a database.
    INPUT:
        sql_string (String): A valid sql-string that describes which data is required.
        db_type (String, "PSQL"): Specifies which database should be called (must be one of ["PSQL", "ACCESS", "SQLITE"]
        catch (Boolean, True): If true, invalid sql-strings will just be ignored, otherwise a crash happens
        verbose (Boolean, False): If true, the status of the operations are printed
    OUTPUT:
        data_frame (Pandas dataframe): The data from the database as a pandas dataframe. If something went wrong and
        catch=True, 'None' will be returned

    - add_data_to_db:
    This function allows to add data to a database.
    INPUT:
        sql_string (String): A valid sql-string that describes which data will be inserted.
        db_type (String, "PSQL"): Specifies which database should be called (must be one of ["PSQL", "ACCESS", "SQLITE"]
        catch (Boolean, True): If true, invalid sql-strings will just be ignored, otherwise a crash happens
        verbose (Boolean, False): If true, the status of the operations are printed
    OUTPUT:
        success (Boolean): Returns 'True' for a successful adding

    - edit_data_in_db:
    This function allows to edit data in a database.
    INPUT:
        sql_string (String): A valid sql-string that describes which data will be edited.
        db_type (String, "PSQL"): Specifies which database should be called (must be one of ["PSQL", "ACCESS", "SQLITE"]
        catch (Boolean, True): If true, invalid sql-strings will just be ignored, otherwise a crash happens
        verbose (Boolean, False): If true, the status of the operations are printed
    OUTPUT:
        success (Boolean): Returns 'True' for a successful editing

    - delete_data_from_db:
    This function allows to delete data from a database.
    INPUT:
        sql_string (String): A valid sql-string that describes which data will be deleted.
        db_type (String, "PSQL"): Specifies which database should be called (must be one of ["PSQL", "ACCESS", "SQLITE"]
        catch (Boolean, True): If true, invalid sql-strings will just be ignored, otherwise a crash happens
        verbose (Boolean, False): If true, the status of the operations are printed
    OUTPUT:
        success (Boolean): Returns 'True' for a successful deleting
"""


def __build_connection(db_type):

    # connection to different databases must be build up different
    if db_type == "PSQL":
        conn = psycopg2.connect(database="<Your db name>",
                                user="<Your user name>",
                                password="<Your password>",
                                host="<Your db adress>"  # Normally stuff like localhost or 127.0.0.1,
                                port="<Your port">)  # Normally a port like 5432
        cursor = conn.cursor()

    elif db_type == "ACCESS":
        # TODO add access
        raise NotImplementedError

    elif db_type == "SQLITE":

        # get path of sql-lite db
        default_sqlite_path = "<The path to your sqllite db>" #  /media/fdahle/beb5a64a-5335-424a-8f3c-779527060523/ATM/data/databases/sqlite/sqlite.db"

        # check if db is up-to-date and if not update the sql-lite file
        ddfd.download_data_from_db()

        conn = sqlite3.connect(default_sqlite_path)
        cursor = conn.cursor()

    else:
        conn = None
        cursor = None

    return conn, cursor


# change the flag so that the local database is download again
# The flag is a simple txt file which the only content being "True" or "False"
def __change_flag():
    default_flag_path = "<The path to your 'flag.txt' file>"

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
def get_data_from_db(sql_string, db_type="PSQL", catch=True, verbose=False):
    assert db_type in ["PSQL", "ACCESS", "FILES"], "The specified database is incorrect"

    if verbose:
        print("Execute '{}' at {}".format(sql_string, db_type))

    # get the connection
    conn, cursor = __build_connection(db_type)

    try:
        data_frame = pd.read_sql(sql_string, conn)
        return data_frame
    except (Exception,) as e:
        if catch:
            return None
        else:
            raise e


# add data to db
def add_data_to_db(sql_string, db_type="PSQL", catch=True, verbose=False):
    assert db_type in ["PSQL", "ACCESS", "FILES"], "The specified database is incorrect"

    if verbose:
        print("Execute '{}' at {}".format(sql_string, db_type))

    # edit and add is the same, but so that people have the possibility to call what they want
    bool_success = edit_data_in_db(sql_string, db_type, catch, verbose=False)

    return bool_success


# change data in db
def edit_data_in_db(sql_string, db_type="PSQL", catch=True, verbose=False):
    assert db_type in ["PSQL", "ACCESS", "FILES"], "The specified database is incorrect"

    if verbose:
        print("Execute '{}' at {}".format(sql_string, db_type))

    # only add last change is not already in sql string, and we are not deleting
    if "last_change" not in sql_string:

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
def delete_data_from_db(sql_string, db_type="PSQL", catch=True, verbose=False):
    assert db_type in ["PSQL", "ACCESS", "FILES"], "The specified database is incorrect"

    if verbose:
        print("Execute '{}' at {}".format(sql_string, db_type))

    # edit and delete is the same, but so that people have the possibility to call what they want
    bool_success = edit_data_in_db(sql_string, db_type, catch, verbose=False)

    return bool_success
