import numpy as np

import base.connect_to_db as ctd
import base.print_v as p


# the segmentation has the following classes:
# 1: ice, 2: snow, 3: rocks, 4: water, 5: clouds, 6:sky, 7: unknown


def update_table_segmented(image_id, segmented, data, overwrite=True,
                           catch=True, verbose=False, pbar=None):
    """
    update_table_segmented(image_id, segmented, data, overwrite, catch, true, verbose, pbar):
    This function calculates the percentage of each class for a segmented image and updates the
    equivalent row in the database.
    Args:
        image_id (String): The image_id of the image that is segmented and that we are updating.
        segmented (np-array): The segmented image
        data (dict): Some additional data for the table (how was the image segmented, which model)
        overwrite: If false, we are checking if the image already has a segmentation and will not update in that case
        catch (Boolean, True): If true and something is going wrong, the operation will continue and not crash.
            In this case None is returned
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar

    Returns:

    """

    p.print_v(f"Start: update_table_segmented ({image_id})", verbose=verbose, pbar=pbar)

    # get the already existing information from the table
    sql_string = f"SELECT * FROM images_segmentation WHERE image_id='{image_id}'"
    table_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)
    table_data = table_data.iloc[0]

    # get the number of unique values per class of the segmented images
    total_number_of_pixels = segmented.shape[0] * segmented.shape[1]
    uniq, counts = np.unique(segmented, return_counts=True)

    # get the percentages from the image & fill the update_dict
    update_dict = {}
    labels = ["perc_ice", "perc_snow", "perc_rocks", "perc_water",
              "perc_clouds", "perc_sky", "perc_other"]

    for i in range(7):

        class_val = i + 1

        # get position of value in uniq
        if class_val in uniq:
            class_idx = np.where(uniq == class_val)[0][0]

            class_count = counts[class_idx]
            update_dict[labels[i]] = (round(class_count / total_number_of_pixels * 100, 2), "float")
        else:
            update_dict[labels[i]] = (0, "float")

    # add more data to the update-dict
    update_dict["labelled_by"] = (data["labelled_by"], "string")
    update_dict["model_name"] = (data["model_name"], "string")

    # create the basis sql-string
    sql_string = "UPDATE images_segmentation SET "

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
        p.print_v(f"No update required for table 'images_segmented'",
                  verbose, pbar=pbar)
        success = True

    # we need to update
    else:
        # remove the last trailing comma
        sql_string = sql_string[:-2]

        # add the condition
        sql_string = sql_string + f" WHERE image_id='{image_id}'"

        # the real updating
        success = ctd.edit_data_in_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

        if success:
            p.print_v(f"Finished: update_table_segmented ({image_id})", verbose, color="green", pbar=pbar)
        else:
            p.print_v(f"Failed: update_table_segmented ({image_id})", verbose, color="red", pbar=pbar)

    return success
