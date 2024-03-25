import os

from tqdm import tqdm

import src.base.connect_to_database as ctd
import src.prepare.create_table_entry as cte

PATH_IMAGE_FLD = "/data_1/ATM/data_1/aerial/TMA/downloaded"

tables_for_adding = ["images"]


def add_images_to_psql(lst_images, tables):

    conn = ctd.establish_connection()

    for table in tables:

        sql_string = f"SELECT image_id FROM {table}"
        data = ctd.execute_sql(sql_string, conn)
        existing_images = data['image_id'].values.tolist()

        # remove existing images from list of images
        lst_images_new = [image for image in lst_images if image not in existing_images]

        for image_id in (pbar := tqdm(lst_images_new)):

            pbar.set_postfix_str(f"Add {image_id} to {table}")
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
