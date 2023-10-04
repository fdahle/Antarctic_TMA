import base.connect_to_db as ctd
import base.print_v as p


def update_table_images_estimated(image_id, update_type, data,
                                  overwrite=True, catch=True, verbose=False, pbar=None):

    # only these four update-types are supported
    assert update_type in ["cam_id", "focal_length", "lens_cone"]

    if update_type == "cam_id":
        update_dict = {"cam_id": (data, "string"),
                       "cam_id_estimated": (True, "boolean")}

    elif update_type == "focal_length":

        update_dict = {"focal_length": (data, "float"),
                       "focal_length_estimated": (True, "boolean")}

    elif update_type == "lens_cone":
        update_dict = {"lens_cone": (data, "string"),
                       "lens_cone_estimated": (True, "boolean")}

    else:
        raise ValueError("update_type not defined")

    # create the basis sql-string
    sql_string = "UPDATE images_extracted SET "

    # iterate all fields to update
    for key in update_dict:  # noqa

        # get the value to update
        val = update_dict[key]

        # check if we add string or other stuff
        if val[1] == "string":
            sql_string = sql_string + key + "='" + str(val[0]) + "', "
        else:
            sql_string = sql_string + key + "=" + str(val[0]) + ", "

    # remove the last trailing comma
    sql_string = sql_string[:-2]

    # add the condition
    sql_string = sql_string + f" WHERE image_id='{image_id}'"

    p.print_v(f"Update table 'images_extracted:{update_type}' for '{image_id}'",
              verbose, pbar=pbar)

    # the real updating
    success = ctd.edit_data_in_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    return success
