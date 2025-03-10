
import pandas as pd
from tqdm import tqdm

import src.base.connect_to_database as ctd

update_types = ["sat", "img", "calc"]
PATH_AZIMUTH_FLD = "/data/ATM/data_1/georef/azimuths/"

def update_azimuths():

    conn = ctd.establish_connection()

    for update_type in update_types:

        # get the path to the azimuths
        path_azimuths = PATH_AZIMUTH_FLD + f"{update_type}_azimuths.csv"

        # load the azimuths
        azimuths = pd.read_csv(path_azimuths)
        azimuths.columns=["image_id", "azimuth"]
        # iterate the file
        for i, row in tqdm(azimuths.iterrows(), total=azimuths.shape[0]):

            # get the image  id
            image_id = row["image_id"]

            # get the azimuth
            azimuth = row["azimuth"]

            # update the database
            sql_string = (f"UPDATE images_georef "
                          f"SET azimuth_exact = {azimuth} "
                          f"WHERE image_id = '{image_id}'")

            ctd.execute_sql(sql_string, conn)

if __name__ == "__main__":
    update_azimuths()