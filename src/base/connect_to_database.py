import pandas as pd
import psycopg2
import warnings

# Global psql constants
PSQL_HOST = "127.0.0.1"
#PSQL_PORT = "7777"
PSQL_PORT = "5432"
PSQL_PASSWORD = "password"
PSQL_USER = "admin"
PSQL_DATABASE = "antarctica2"

def establish_connection():
    return psycopg2.connect(
        database=PSQL_DATABASE,
        user=PSQL_USER,
        password=PSQL_PASSWORD,
        host=PSQL_HOST,
        port=PSQL_PORT
    )

def execute_sql(sql_string, conn):

    # get cursor for this connection
    cursor = conn.cursor()

    # Determine sql query type based on the start of the sql_string
    action_type = sql_string.strip().split()[0].upper()

    # we want to get data from db        
    if action_type == "SELECT":

        # ignore the pandas sqlAlchemy warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # use pandas to get the data from database
            data_frame = pd.read_sql(sql_string, conn)

        return data_frame

    # we want to edit data in db
    else:
        
        # execute the edit
        cursor.execute(sql_string)
        conn.commit()