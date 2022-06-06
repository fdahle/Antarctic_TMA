import random

import get_ids_from_folder as giff
import load_image_from_file as liff
import extract_fid_marks as efm
import connect_to_db as ctd

input_path = "<Please enter the path where your images are located>"

"""
This function is used to extract the fid poitns for all images and save it in a database.
Please note that to extract fid points subsets must be extracted before.
If overwrite is false, existing subsets are not overwritten and instead ignored for saving in the db
"""

def apply_mass_fid_point_extraction(path_folder_images, overwrite=False, shuffle=False, catch=True):

    # get ids from the folder
    ids = giff.get_ids_from_folder([path_folder_images])

    # random shuffle
    if shuffle:
        random.shuffle(ids)

    # get total number of ids
    total_number = len(ids)

    for i, img_id in enumerate(ids):

        perc = round((i+1)/total_number * 100, 2)

        print(f"\rextract fid marks for {img_id} - {i+1}/{total_number} ({perc}%)", end='')

        try:

            sql_string = "SELECT * FROM images_properties WHERE image_id='" + img_id + "'"
            data = ctd.get_data_from_db(sql_string).iloc[0]

            # if all subsets are missing we cannot calculate the fid points
            if data["subset_n_x"] is None and data["subset_e_x"] is None and \
                    data["subset_s_x"] is None and data["subset_w_x"] is None:
                continue

            # if all fid points are already there and overwrite is false also no need to continue
            skip_arr = []
            if overwrite is False:
                skip_image = True
                for j in range(8):
                    if data[f"fid_mark_{j+1}_y"] is not None and data[f"fid_mark_{j+1}_y"] is not None:
                        skip_arr.append(True)
                    else:
                        skip_arr.append(False)
                        skip_image = False
                if skip_image:
                    continue

            else:
                for j in range(8):
                    skip_arr.append(False)

            img = liff.load_image_from_file(img_id)

            fid_points = efm.extract_fid_marks(img, data)

            # even though I name fid points based on their sky direction, fid points have a unique number
            # see for example here: https://data.pgc.umn.edu/aerial/usgs/tma/docs/photoinfo_1983.pdf
            conversion_dict = {
                "n": 7,
                "e": 6,
                "s": 8,
                "w": 5,
                "ne":2,
                "se":4,
                "nw":1,
                "sw":3
            }

            sql_string = "UPDATE images_properties SET "
            for j, key in enumerate(fid_points):

                # skip this
                if skip_arr[j]:
                    continue

                val = fid_points[key]

                if val is None:
                    continue

                nbr = conversion_dict[key]
                sql_string = sql_string + f"fid_mark_{nbr}_x={val[0]}, "
                sql_string = sql_string + f"fid_mark_{nbr}_y={val[1]}, "
                sql_string = sql_string + f"fid_mark_{nbr}_estimated=False, "

            sql_string = sql_string + f"fid_extraction_date=NOW() WHERE image_id='{img_id}'"

            ctd.edit_data_in_db(sql_string, catch=False)

        except (Exception,) as e:
            if catch:
                continue
            else:
                raise e


if __name__ == "__main__":
    apply_mass_fid_point_extraction(input_path)
