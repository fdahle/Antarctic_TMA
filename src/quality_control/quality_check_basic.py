import glob
import json
import os

import base.connect_to_db as ctd
import base.print_v as p

debug_more_details = True
debug_max_cutoff = 100


def quality_check_basic(base_folder=None, folders=None, tables=None):
    """
    quality_check_basic (base_folder, folders, tables):
    This function performs a basic quality check by checking the following:
     - Are there the same files in all provided folders
     - Are there any other files than tiff-files in the provided folders
     - Are the same ids in all provided tables
     - Are there ids in the table that are not in the folders
     - Are there files in the folders that are not in the tables
    Args:
        base_folder (String): The path to the folder in which all other folders are
        folders (List): The names of the folder in base_folder
        tables (List): The names of the tables we are checking
    Returns:
        None
    """

    print("Start quality check: basic")

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if base_folder is None:
        base_folder = json_data["path_folder_base"]

    if folders is None:
        folders = json_data["check_basic_folders"]

    if tables is None:
        tables = json_data["check_basic_tables"]

    # here we save all ids per folder
    dict_ids_per_folder = {}

    # get the tif-files in the folders
    for _folder in folders:
        dict_ids_per_folder[_folder] = glob.glob1(base_folder + "/" + _folder, "*.*")

    # iterate all folders
    for i, elem in enumerate(folders):

        # save ids in folder (and remove possible OIS-Reech)
        base_list = []
        for filename in dict_ids_per_folder[elem]:
            if filename.startswith("OIS-Reech_"):
                base_list.append(filename[10:])
            else:
                base_list.append(filename)

        # iterate the other folders
        for j, _elem in enumerate(folders):

            # we don't need to compare the same folder twice
            if j == i:
                continue

            # save the other ids in folder
            compare_list = []
            for elem2 in dict_ids_per_folder[_elem]:
                if elem2.startswith("OIS-Reech_"):
                    compare_list.append(elem2[10:])
                else:
                    compare_list.append(elem2)

            # difference
            differences = list(set(base_list) - set(compare_list))

            if len(differences) == 0:
                p.print_v(f"No images are missing in folder '{elem}' "
                          f"based on folder '{_elem}'", color="green")
            else:
                p.print_v(f"There are {len(differences)} images in "
                          f"folder '{elem}' that are not in "
                          f"folder '{_elem}'", color="red")
                if debug_more_details:
                    if len(differences) > debug_max_cutoff:
                        p.print_v(differences[0:debug_max_cutoff])
                        p.print_v(f"list was cut off after {debug_max_cutoff} entries")
                    else:
                        p.print_v(differences)


    # here we save all ids per table
    dict_table_ids = {}

    # get the table data
    for table in tables:
        sql_string = f"SELECT image_id FROM {table}"
        data = ctd.get_data_from_db(sql_string).values.tolist()

        # flatten list
        data = [item for sublist in data for item in sublist]

        dict_table_ids[table] = data

    # iterate all tables
    for i, elem in enumerate(tables):

        base_list = dict_table_ids[elem]

        # iterate all sub-tables again
        for j, _elem in enumerate(tables):

            if j == i:
                continue
            compare_list = dict_table_ids[_elem]

            # difference
            differences = list(set(base_list) - set(compare_list))

            if len(differences) == 0:
                p.print_v(f"No images are missing in table '{_elem}' "
                          f"based on table '{elem}'", color="green")
            else:
                p.print_v(f"There are {len(differences)} images in table '{elem}' that are not in "
                          f"table '{_elem}'", color="red")
                if debug_more_details:
                    if len(differences) > debug_max_cutoff:
                        p.print_v(differences[0:debug_max_cutoff])
                        p.print_v(f"list was cut off after {debug_max_cutoff} entries")
                    else:
                        p.print_v(differences)

    # iterate all folders and tables
    for i, elem in enumerate(folders):

        # save ids in folder (and remove possible OIS-Reech)
        base_list = []
        for _elem in dict_ids_per_folder[elem]:
            _elem = _elem[:-4]
            if _elem.startswith("OIS-Reech_"):
                base_list.append(_elem[10:])
            else:
                base_list.append(_elem)

        for j, _elem in enumerate(tables):

            compare_list = dict_table_ids[_elem]

            # difference
            differences = list(set(base_list) - set(compare_list))

            if len(differences) == 0:
                p.print_v(f"No images are missing in folder '{elem}' "
                          f"based on table '{_elem}", color="green")
            else:
                p.print_v(f"There are {len(differences)} images in "
                          f"folder '{elem}' that are not in "
                          f"table '{_elem}'", color="red")
                if debug_more_details:
                    if debug_more_details:
                        if len(differences) > debug_max_cutoff:
                            p.print_v(differences[0:debug_max_cutoff])
                            p.print_v(f"list was cut off after {debug_max_cutoff} entries")
                        else:
                            p.print_v(differences)

    # iterate all tables and folder
    for i, elem in enumerate(tables):

        base_list = dict_table_ids[elem]

        for j, _elem in enumerate(folders):

            # save ids in folder (and remove possible OIS-Reech)
            compare_list = []
            for elem2 in dict_ids_per_folder[_elem]:
                elem2 = elem2[:-4]
                if elem2.startswith("OIS-Reech_"):
                    compare_list.append(elem2[10:])
                else:
                    compare_list.append(elem2)

            # difference
            differences = list(set(base_list) - set(compare_list))

            if len(differences) == 0:
                p.print_v(f"No images are missing in table '{elem}' "
                          f"based on folder '{_elem}", color="green")
            else:
                p.print_v(f"There are {len(differences)} images in "
                          f"table '{elem}' that are not in "
                          f"folder '{_elem}'", color="red")
                if debug_more_details:
                    if debug_more_details:
                        if len(differences) > debug_max_cutoff:
                            p.print_v(differences[0:debug_max_cutoff])
                            p.print_v(f"list was cut off after {debug_max_cutoff} entries")
                        else:
                            p.print_v(differences)

    # check for non-tiff files
    for elem in folders:
        false_files = []
        for filename in dict_ids_per_folder[elem]:
            if filename.endswith(".tif") is False:
                false_files.append(filename)

        if len(false_files) == 0:
            p.print_v(f"No foreign files are in folder '{elem}'", color="green")
        else:
            p.print_v(f"There are {len(false_files)} foreign files in folder '{elem}'", color="red")
            if debug_more_details:
                if debug_more_details:
                    if len(false_files) > debug_max_cutoff:
                        p.print_v(false_files[0:debug_max_cutoff])
                        p.print_v(f"list was cut off after {debug_max_cutoff} entries")
                    else:
                        p.print_v(false_files)

    # check if images have the same size in different folders


if __name__ == "__main__":
    quality_check_basic()
