import sqlite3

from typing import Any, Optional

import src.base.connect_to_database as ctd


def create_table_entry(image_id: str, table: str,
                       value_dict: Optional[dict[str, Any]] = None,
                       conn: Optional[sqlite3.Connection] = None) -> None:
    """
    Creates an entry in a specified table for a given image ID, with optional additional values.

    Args:
        image_id (str): The unique identifier for the image to be added to the database.
        table (str): The name of the table where the image ID and optional values should be inserted.
        value_dict (Optional[Dict[str, Any]]): A dictionary containing additional column-value
            pairs to be inserted along with the image ID. Defaults to None.
        conn (Optional[sqlite3.Connection]): An existing database connection.
            If None, a new connection will be established. Defaults to None.

    Returns:
        None: This function does not return anything. It either inserts the new
            entry into the database or prints a message if the image ID is already present.
    """

    # create connection to the database if not existing
    if conn is None:
        conn = ctd.establish_connection()

    # check if the image_id is already in the database
    sql_string = f"SELECT * FROM {table} WHERE image_id = '{image_id}'"
    data = ctd.execute_sql(sql_string, conn)
    if data.shape[0] > 0:
        return

    # insert the image_id into the database
    sql_string = f"INSERT INTO images (image_id) VALUES ('{image_id}')"

    # add optionally the values from the value_dict
    if value_dict is not None:
        sql_string = f"INSERT INTO {table} (image_id, "
        for key, value in value_dict.items():
            sql_string += f"{key}, "
        sql_string = sql_string[:-2] + ") VALUES ('{image_id}', "
        for key, value in value_dict.items():
            sql_string += f"{value}, "
        sql_string = sql_string[:-2] + ")"

    # insert the image into the database
    ctd.execute_sql(sql_string, conn)
