import base.connect_to_db as ctd
import base.print_v as p


def update_table_images_extracted(image_id, update_type, data,
                                  overwrite=True, catch=True, verbose=False, pbar=None):

    # only these four update-types are supported
    assert update_type in ["altimeter", "circle", "height", "text", "params", "complexity"]

    # get the already existing information from the table
    sql_string = f"SELECT * FROM images_extracted WHERE image_id='{image_id}'"
    table_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)
    table_data = table_data.iloc[0]

    # update the altimeter
    if update_type == "altimeter":

        # get the right altimeter data from the data
        altimeter_x = data[0]
        altimeter_y = data[2]
        altimeter_width = data[1] - data[0]
        altimeter_height = data[3] - data[2]

        update_dict = {
            "altimeter_x": [altimeter_x, "integer"],
            "altimeter_y": [altimeter_y, "integer"],
            "altimeter_width": [altimeter_width, "integer"],
            "altimeter_height": [altimeter_height, "integer"]
        }

    # update the circle
    elif update_type == "circle":

        # convert the data to a string
        circle_str = str(data[0]) + "," + str(data[1]) + "," + str(data[2])

        update_dict = {
            "circle_pos": [circle_str, "string"]
        }

    # update the height
    elif update_type == "height":
        update_dict = {
            "height": [data, "integer"],
            "height_estimated": [False, "boolean"]
        }

    # update the text
    elif update_type == "text":

        # get text and location from the data dict again
        text = data["text"],
        text_bounds = data["text_bounds"]

        # somehow these variables are in tuples, so remove the tuple
        text = text[0]

        # remove all ; from the text, as we need this to divide the text
        text_adapted = []
        for elem in text:
            elem_adapted = elem.replace(";", "")
            text_adapted.append(elem_adapted)

        # remove all ' from the text, as this disturbs sql
        text_adapted = []
        for elem in text:
            elem_adapted = elem.replace("'", "")
            text_adapted.append(elem_adapted)

        # textbox is [min_x_abs, min_y_abs, max_x_abs, max_y_abs]

        # create a string for each bounding box
        text_bounds_adapted = []
        for elem in text_bounds:
            elem_adapted = "("
            for _elem in elem.bounds:
                elem_adapted = elem_adapted + str(int(_elem)) + ","
            elem_adapted = elem_adapted[:-1] + ")"
            text_bounds_adapted.append(elem_adapted)

        # convert the lists to a string
        text_string = ';'.join(text_adapted)
        text_bounds_string = ';'.join(text_bounds_adapted)

        update_dict = {
            "text_bbox": [text_bounds_string, "string"],
            "text_content": [text_string, "string"]
        }

    elif update_type == "params":

        update_dict = {}
        for key in data:
            if data[key] is not None:
                if key in ["focal_length", "awar"]:
                    d_type = "float"
                elif key in ["height"]:
                    d_type = "integer"
                elif key in ["cam_id", "lens_cone"]:
                    d_type = "string"
                else:
                    raise ValueError("Wrong data-key")
                update_dict[key] = (data[key], d_type)
                update_dict[key + "_estimated"] = (False, "boolean")

    elif update_type == "complexity":
        update_dict = {
            "complexity": [data, "float"],
        }

    else:
        raise ValueError("update_type not defined")

    # create the basis sql-string
    sql_string = "UPDATE images_extracted SET "

    # check how many entries we need to update
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
