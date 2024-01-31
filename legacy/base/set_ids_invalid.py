import json
import os

import base.connect_to_db as ctd

delete_folders = ["downloaded", "masked", "segmented"]
databases = ["images", "images_extracted", "images_fid_points",
             "images_segmentation", "images_tie_points"]


def set_ids_invalid(image_ids, reason):

    # ask question if you're sure
    confirmation = input("Are you sure you want to continue? (yes/no): ")

    if confirmation.lower() != "yes":
        print("Stopped setting ids to invalid")
        return

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    base_path = json_data["path_folder_base"]

    # do operations for every image image_id
    for image_id in image_ids:

        # delete from all folders
        for fld in delete_folders:

            # get the exact path of the file
            file_path = base_path + fld + "/" + image_id + ".tif"

            # check if exists and then delete
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File '{file_path}' deleted successfully.")
            else:
                print(f"File '{file_path}' does not exist.")

        # delete from all databases
        for db in databases:

            # standard case for deleting
            sql_string = f"DELETE FROM {db} where image_id='{image_id}'"

            # special case for images_tie_points
            if db == "images_tie_points":
                sql_string = f"DELETE FROM {db} WHERE image_1_id ='{image_id}' " \
                             f"OR image_2_id='{image_id}'"

            ctd.delete_data_from_db(sql_string, catch=False)
            print(f"Entry deleted from {db}")

        # save in the invalid table
        sql_string = f"INSERT INTO images_invalid (image_id, reason) VALUES " \
                     f"('{image_id}', '{reason}')"
        success = ctd.add_data_to_db(sql_string)

        if success:
            print(f"Entry created in failed database for {image_id}")
        else:
            print(f"Entry failed for '{image_id}'")

if __name__ == "__main__":

    img_ids = ['CA172031L0258']

    set_ids_invalid(img_ids, "invalid_tiff")