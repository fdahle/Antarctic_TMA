"""manages connection to the database"""

# Library imports
import pandas as pd
import psycopg2
import warnings
from datetime import datetime, timezone
from typing import Optional

# Global psql constants
PSQL_HOST = "127.0.0.1"
# PSQL_PORT = "7777"
PSQL_PORT = "5432"
PSQL_PASSWORD = "password"
PSQL_USER = "postgres"
PSQL_DATABASE = "tma_db"


def establish_connection() -> psycopg2.extensions.connection:
    """
    Establishes a connection to the PostgresSQL database using the global constants.
    Returns:
        psycopg2.extensions.connection: The connection object.
    """
    return psycopg2.connect(
        database=PSQL_DATABASE,
        user=PSQL_USER,
        password=PSQL_PASSWORD,
        host=PSQL_HOST,
        port=PSQL_PORT
    )


def execute_sql(sql_string: str,
                conn: psycopg2.extensions.connection,
                add_timestamp: bool = True) \
        -> Optional[pd.DataFrame]:
    """
    Executes an SQL string using a given database connection and returns the data as a pandas
    DataFrame for SELECT queries, or commits the action for non-SELECT queries without
    returning data. Optionally, updates 'last_change' column with the current timestamp
    when adding or editing a row.

    Args:
        sql_string (str): The SQL query string to be executed.
        conn (psycopg2.extensions.connection): The database connection object.
        add_timestamp (bool): If True, automatically update 'last_change' with the
            current timestamp on insert or update.

    Returns:
        Optional[pd.DataFrame]: A pandas DataFrame containing the result of the SELECT query, or None
        for non-SELECT queries.
    """

    # get cursor for this connection
    cursor = conn.cursor()

    # Determine sql query type based on the start of the sql_string
    action_type = sql_string.strip().split()[0].upper()

    # Check if we need to add a timestamp and the action is not a SELECT query
    if add_timestamp and action_type in ("INSERT", "UPDATE"):

        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        if action_type == "INSERT":
            # Split the INSERT statement to correctly insert the timestamp
            parts = sql_string.strip().split(') VALUES')
            columns_part = parts[0] + ', last_change)'
            values_part = 'VALUES' + parts[1].strip()

            # Insert the timestamp into the VALUES part
            values_part = values_part[:-1] + f", '{timestamp}')"  # Assuming there's always a closing parenthesis
            sql_string = columns_part + ' ' + values_part

        elif action_type == "UPDATE":
            # Insert last_change update before WHERE clause, or at the end if no WHERE clause
            where_index = sql_string.upper().find(' WHERE ')
            if where_index != -1:
                # Insert the timestamp update before WHERE clause
                sql_string = sql_string[:where_index] + f", last_change = '{timestamp}'" + sql_string[where_index:]
            else:
                # Append the timestamp update at the end of the statement
                sql_string += f", last_change = '{timestamp}'"

    # we want to get data from db        
    if action_type == "SELECT":

        # ignore the pandas sqlAlchemy warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # use pandas to get the data from database
            data_frame = pd.read_sql(sql_string, conn)

        return data_frame

    # we want to edit/delete data in db
    else:

        # execute the edit
        try:
            cursor.execute(sql_string)
            conn.commit()
        except (Exception,) as e:
            print(sql_string)
            print(e)
            raise e
        return None
