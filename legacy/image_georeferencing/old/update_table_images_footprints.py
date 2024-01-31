import copy

import base.connect_to_db as ctd
import base.print_v as p

def update_table_images_footprints(image_id, update_type, data, overwrite=True,
                                   catch=True, verbose=False, pbar=None):

    # copy overwrite to only change it in here
    overwrite = copy.deepcopy(overwrite)

    if overwrite is False:
        sql_string = f"SELECT * FROM images_extracted WHERE image_id='{image_id}'"
        table_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

        # if we don't have any data for this image-image_id we can set overwrite to true
        if table_data.shape[0] == 0:
            overwrite = True

        # extract the data from the dataframe
        table_data = table_data.iloc[0]
    else:
        table_data = None

    if update_type == "footprint_approx":
        footprint_approx = data

        update_dict = {
            "footprint_approx": [footprint_approx, "geom"]
        }

        # create the basis sql-string
        sql_string = "UPDATE images_extracted SET "

    nr_updated_entries = 0

    # iterate all fields to update
    for key in update_dict:  # noqa

        # check if we need to update
        if overwrite is True or table_data[key] is None:

            # increment update counter
            nr_updated_entries += 1

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
            elif val[1] == 'geom':
                sql_string = sql_string + key + "='" + str(val[0]) + "', "
            else:
                sql_string = sql_string + key + "=" + str(val[0]) + ", "

    # nothing to update
    if nr_updated_entries == 0:
        p.print_v(f"No update required for table 'images_extracted:{update_type}'",
                  verbose, pbar=pbar)
        return True

    # remove the last trailing comma
    sql_string = sql_string[:-2]

    # add the condition
    sql_string = sql_string + f" WHERE image_id='{image_id}'"

    p.print_v(f"Update table 'images_extracted:{update_type}' for '{image_id}'",
              verbose, pbar=pbar)

    # the real updating
    success = ctd.edit_data_in_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    return success
