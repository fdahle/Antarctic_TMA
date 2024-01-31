import json
import os

import base.connect_to_db as ctd
import base.print_v as p


def create_image_entry_in_database(image_id, databases=None, overwrite=True,
                                   catch=True, verbose=False, pbar=None):
    """
    create_image_entry_in_database(input_img, image_id, catch, verbose, display):
    This function adds an image entry to all specified databases
    Args:
        image_id (String): The image image_id of image that should be added to the database.
        databases (List): A list containing all databases in which we want to add an entry
        overwrite (Boolean): If true and the entry is already existing we can skip this function
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        status (Boolean): True if everything worked, False if something went wrong
    """

    p.print_v(f"Start: create_image_entry_in_database ({image_id})", verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if databases is None:
        databases = json_data["create_image_entry_databases"]

    db_success = []

    for database in databases:

        # create sql string
        sql_string = f"SELECT image_id from {database} WHERE image_id ='{image_id}'"

        # first check if the image is in the database
        data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

        # data already there
        if data.shape[0] == 1:
            if overwrite is False:
                p.print_v(f"{image_id} already existing in table '{database}'", verbose, "green", pbar=pbar)
                db_success.append(True)
                continue
            else:
                # we want a new start, so delete this entry so that we can add it to the db again
                sql_string = f"DELETE FROM {database} WHERE image_id='{image_id}'"
                ctd.delete_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

        # this means something went wrong in our db
        elif data.shape[0] > 1:
            p.print_v(f"Something went wrong with {image_id}, as it is multiple times in"
                      f" '{database}'. Please check your data", verbose, color="red", pbar=pbar)
            db_success.append(False)
            if catch is False:
                exit()
            else:
                continue

        # create insert sql string for 'images'
        sql_string = f"INSERT INTO {database} (image_id) VALUES ('{image_id}')"
        success = ctd.add_data_to_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

        # check for success
        if success is False:
            if catch:
                db_success.append(False)
                continue
            else:
                raise Exception(f"Something went wrong adding {image_id} to {database}")

        db_success.append(True)

        p.print_v(f"Finished: create_image_entry_in_database ({image_id})", verbose, pbar=pbar)

    return all(db_success)
