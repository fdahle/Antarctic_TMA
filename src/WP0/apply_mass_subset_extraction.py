import random

import get_ids_from_folder as giff
import load_image_from_file as liff
import extract_subsets as es
import connect_to_db as ctd

input_path = "<Please enter the path where your images are located>"

"""
This function is used to extract subsets for all images and save it in a database.
If overwrite is false, existing subsets are not overwritten and instead ignored for saving in the db
"""


def apply_mass_subset_extraction(path_folder_images, overwrite=False, shuffle=False, catch=True):

    # get ids from the folder
    ids = giff.get_ids_from_folder([path_folder_images])

    # random shuffle
    if shuffle:
        random.shuffle(ids)

    # get total number of ids
    total_number = len(ids)

    for i, img_id in enumerate(ids):

        perc = round((i+1)/total_number * 100, 2)

        print(f"\rextract subsets for {img_id} - {i+1}/{total_number} ({perc}%)", end='')

        try:

            if overwrite is False:
                sql_string = "SELECT subset_n_x, subset_n_y, subset_e_x, subset_e_y, " \
                             "subset_s_x, subset_s_y, subset_w_x, subset_w_y FROM images_properties " \
                             "WHERE image_id='" + img_id + "'"

                data = ctd.get_data_from_db(sql_string).iloc[0]

                # if all coordinates are already there -> no need to do the rest
                if data["subset_n_x"] is not None and data["subset_e_x"] is not None and \
                   data["subset_s_x"] is not None and data["subset_w_x"] is not None:
                    continue
            else:
                # not required, but the variable is initialized so that the ide is not complaining
                data = None

            img = liff.load_image_from_file(img_id)

            # here the actual extraction is happening
            coordinates = es.extract_subsets(img)

            # if all coordinates are None we can skip this image
            if coordinates["n"] is None and coordinates["e"] is None and \
               coordinates["s"] is None and coordinates["w"] is None:
                continue

            sql_string = "UPDATE images_properties SET subset_height=250, subset_width=250, "

            for direction in ["n", "e", "s", "w"]:
                if coordinates[direction] is not None:
                    if overwrite is False and data[f"subset_{direction}_x"] is None:
                        sql_string = sql_string + f"subset_{direction}_x={str(coordinates["n"][0])}, "
                        sql_string = sql_string + f"subset_{direction}_y={str(coordinates["n"][2])}, "

            sql_string = sql_string + ", subset_extraction_date=NOW() WHERE image_id='" + img_id + "'"

            ctd.edit_data_in_db(sql_string)

        except (Exception,) as e:
            if catch:
                pass
            else:
                raise e


if __name__ == "__main__":
    apply_mass_subset_extraction(input_path)
