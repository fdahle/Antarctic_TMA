# Library imports
import os
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.prepare.create_table_entry as cte

# Constants
PATH_IMAGE_FLD = "/data/ATM/data_1/aerial/TMA/downloaded_child"

# Variables
tables_for_adding = ["images", "images_extracted", "images_fid_points", "images_georef"]


def add_images_to_psql(lst_images: list[str], tables: list[str]) -> None:
    """
    Adds images to PostgreSQL database tables if they don't already exist.
    Args:
        lst_images: A list of image names (strings) to be added.
        tables: A list of table names (strings) in the database to add the images to.
    """

    # establish connection to psql
    conn = ctd.establish_connection()

    # iterate over all tables
    for table in tables:

        # get all existing images from the table
        sql_string = f"SELECT image_id FROM {table}"
        data = ctd.execute_sql(sql_string, conn)
        existing_images = data['image_id'].values.tolist()

        # remove existing images from list of images
        lst_images_new = [image for image in lst_images if image not in existing_images]

        # iterate over all images
        for image_id in (pbar := tqdm(lst_images_new)):

            # update the progress bar
            pbar.set_postfix_str(f"Add {image_id} to {table}")

            # add image to psql
            cte.create_table_entry(image_id, table, conn=conn)


if __name__ == "__main__":

    # get all files in the folder
    all_files = os.listdir(PATH_IMAGE_FLD)

    # only keep tif files
    all_files = [file for file in all_files if file.endswith(".tif")]

    # remove ending
    all_files = [file[:-4] for file in all_files]

    # call function to add images to psql
    add_images_to_psql(all_files, tables_for_adding)
