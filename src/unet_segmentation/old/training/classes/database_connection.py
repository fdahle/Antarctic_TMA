import json
import pandas as pd
import pathlib
import os
import sqlite3

param_path = str(pathlib.Path(__file__).parent.resolve())
with open(param_path + '/' + 'params_public.json') as json_file:
    params_public = json.load(json_file)


class Connector:

    def __init__(self, catch=True):

        self.catch = catch

        real_path = os.path.realpath(__file__)
        atm_path = real_path[:-35]

        if params_public["db_conn"] == "files":

            db_folder= atm_path + "/" + "data/databases/files"
            db_path = db_folder + "/database.db"

            # create db if not existing
            if os.path.isfile(db_path) == False:

                # create conn if not available
                self.conn = sqlite3.connect(db_path)

                for filename in os.listdir(db_folder):
                    if filename.endswith(".csv"):

                        file_id = filename.split(".")[0]

                        # load csv
                        df = pd.read_csv(db_folder + "/" + file_id + ".csv", sep=";")

                        print(df.head())

                        # store your table in the database:
                        df.to_sql(file_id, self.conn)

            else:
                self.conn = sqlite3.connect(db_path)

    def count_data(self, sql_string):

        if self.catch:
            try:
                self.cursor.execute(sql_string)

                count = self.cursor.fetchall()[0][0]

                return count
            except (Exception,):
                return None
        else:
            self.cursor.execute(sql_string)
            count = self.cursor.fetchall()[0][0]
            return count

    def get_data(self, sql_string):

        if self.catch:
            try:
                data_frame = pd.read_sql(sql_string, self.conn)
                return data_frame
            except (Exception,):
                return None
        else:
            data_frame = pd.read_sql(sql_string, self.conn)
            return data_frame

    def add_data(self, sql_string):

        # edit and add is the same, but so that people have the possibility to call what they want
        bool_success = self.edit_data(sql_string)

        return bool_success

    def edit_data(self, sql_string):

        # in psql date() is now()
        if params_public["db_conn"] == "PSQL":
            sql_string = sql_string.replace("=Date()", "=NOW()")

        if self.catch:
            try:
                self.cursor.execute(sql_string)
                self.conn.commit()
                return True
            except (Exception,):
                return False
        else:
            self.cursor.execute(sql_string)
            self.conn.commit()
            return True
