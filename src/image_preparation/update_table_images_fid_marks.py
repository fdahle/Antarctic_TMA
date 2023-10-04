import copy

import base.connect_to_db as ctd
import base.print_v as p


def update_table_images_fid_marks(image_id, update_type, data,
                                  overwrite=False, catch=True, verbose=False, pbar=None):
    """
    update_table_images_fid_marks(image_id, update_type, data, overwrite, catch, verbose, pbar):
    Update the table images_fid_marks with the subset and fid mark data
    Args:
        image_id (str): The image_id of the entry we want to update
        update_type (str): what type of data do we want to update ('basic', 'subset', 'fid_marks')
        data (pandas): The dataframe we are using to update
        overwrite (Boolean): If this is true, we overwrite regardless of existing data
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        success (Boolean): If this is false, something went wrong during execution
    """

    assert update_type in ["basic", "subsets", "fid_marks"]

    # copy overwrite to only change it in here
    overwrite = copy.deepcopy(overwrite)

    if overwrite is False:
        sql_string = f"SELECT * FROM images_fid_points WHERE image_id='{image_id}'"
        table_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

        # if we don't have any data for this image-image_id we can set overwrite to true
        if table_data.shape[0] == 0:
            overwrite = True

        # extract the data from the dataframe
        table_data = table_data.iloc[0]
    else:
        table_data = None

    if update_type == "basic":

        # save all params for updating
        update_dict = {
            "image_width": [data["image_width"], "integer"],
            "image_height": [data["image_height"], "integer"]
        }

    # we want to update subsets
    elif update_type == "subsets":

        # save all params for updating
        update_dict = {
            "subset_height": [250, 'integer'],
            "subset_width": [250, 'integer']
        }

        # iterate all keys
        for key in ["n", "e", "s", "w"]:

            # if we don't have data we can continue
            if data[key] is None:
                continue

            update_dict[f"subset_{key}_x"] = [data[key][0], "integer"]
            update_dict[f"subset_{key}_y"] = [data[key][2], "integer"]
            update_dict[f"subset_{key}_estimated"] = ["False", "boolean"]
            update_dict[f"subset_{key}_extraction_date"] = ["NOW()", "TIMESTAMP"]

    elif update_type == "fid_marks":

        update_dict = {}

        # iterate all keys
        for key in [1, 2, 3, 4, 5, 6, 7, 8]:

            # if we don't have data we can continue
            if data[key] is None:
                continue

            update_dict[f"fid_mark_{key}_x"] = [data[key][0], "integer"]
            update_dict[f"fid_mark_{key}_y"] = [data[key][1], "integer"]
            update_dict[f"fid_mark_{key}_estimated"] = ["False", "boolean"]
            update_dict[f"fid_mark_{key}_extraction_date"] = ["NOW()", "TIMESTAMP"]

    else:
        update_dict = {}

    # create the basis sql-string
    sql_string = "UPDATE images_fid_points SET "

    # check how many entries we need to update
    nr_updated_entries = 0
    timestamp_counter = 0

    # iterate all fields to update
    for key in update_dict:  # noqa

        # check if we need to update
        if overwrite is True or table_data[key] is None:

            # increment update counter
            nr_updated_entries += 1
            timestamp_counter += 1

            # get the value to update
            val = update_dict[key]

            # we don't need to update empty values
            if val[0] is None:
                continue

            # check if we add string or other stuff
            if val[1] == "string":
                sql_string = sql_string + key + "='" + str(val[0]) + "', "
            elif val[1] == "TIMESTAMP":
                if timestamp_counter > 0:
                    timestamp_counter = 0
                    sql_string = sql_string + key + "='" + str(val[0]) + "', "
            else:
                sql_string = sql_string + key + "=" + str(val[0]) + ", "

    # nothing to update
    if nr_updated_entries == 0:
        p.print_v(f"No update required for table 'images_fid_marks:{update_type}'",
                  verbose, pbar=pbar)
        return True

    # remove the last trailing comma
    sql_string = sql_string[:-2]

    # add the condition
    sql_string = sql_string + f" WHERE image_id='{image_id}'"

    p.print_v(f"Update table 'images_fid_marks:{update_type}' for '{image_id}'",
              verbose, pbar=pbar)

    # the real updating
    success = ctd.edit_data_in_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    return success
