import copy

import base.connect_to_db as ctd
import base.print_v as p


def update_table_images(image_id, image_path, shape_data, overwrite=True,
                        catch=True, verbose=False, pbar=None):
    """
    update_table_images(image_id, image_path, shape_data, overwrite, catch, verbose, pbar)
    update the table images with the data of the shape file
    Args:
        image_id (str): The image_id of the entry we want to update
        image_path (str): The path to the image
        shape_data (pandas): The other data for this image
        overwrite (Boolean): If this is true, we overwrite regardless of existing data
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        success (Boolean): If this is false, something went wrong during execution
    """

    # copy overwrite to only change it in here
    overwrite = copy.deepcopy(overwrite)

    if overwrite is False:
        sql_string = f"SELECT * FROM images WHERE image_id='{image_id}'"
        table_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

        # if we don't have any data for this image-image_id we can set overwrite to true
        if table_data.shape[0] == 0:
            overwrite = True

        # extract the data from the dataframe
        table_data = table_data.iloc[0]
    else:
        table_data = None

    # get the entity image_id from the image image_id (it's almost the same,
    # just without the view_direction)
    entity_id = image_id[:6] + image_id[-4:]

    # get the row for the image from the shape_data
    try:
        shape_row = shape_data[shape_data["ENTITY_ID"] == entity_id].to_dict(orient='records')[0]
    except (Exception,) as e:
        if catch:
            return False
        else:
            raise e

    # get the image params from the image_id
    tma = image_id[2:6]
    roll = image_id[6:9]
    frame = int(image_id[9:13])
    view_direction = image_id[8]
    altitude = float(shape_row["Altitude"])
    azimuth = float(shape_row["Azimuth_dd"])
    date_of_recording = str(shape_row["Date_"]).rstrip('\n')
    cam_id = int(shape_row["v_cam"])
    location = str(shape_row["Location"])
    point_x = float(shape_row["POINT_X"])
    point_y = float(shape_row["POINT_Y"])
    point = f"ST_GeomFromText('POINT({point_x} {point_y})',3031)"

    # get the geometry
    geom = str(shape_row["geometry"])
    x_coords = geom[7:].split(" ")[0]
    y_coords = geom[:-1].split(" ")[2]

    # fallback if the date cannot be recognized
    date_year = 0
    date_month = 0
    date_day = 0

    # the date_ field is something messy, so we need different methods to handle it
    if len(date_of_recording) == 4:  # only year available
        date_year = int(date_of_recording)
        date_month = 0
        date_day = 0
    elif date_of_recording.count('/') == 2:  # we have all things
        date_year = int(date_of_recording.split("/")[2])
        date_month = int(date_of_recording.split("/")[0])
        date_day = int(date_of_recording.split("/")[1])
    elif date_of_recording.count('/') == 1:
        # super strange combi with two years shown in date
        if date_of_recording[4] == "/" and \
                len(date_of_recording.split("/")[0]) == 4 and \
                len(date_of_recording.split("/")[1]) == 2:
            date_year = int(date_of_recording.split("/")[0])
            date_month = 0
            date_day = 0
        # only month and year
        if len(date_of_recording.split("/")[1]) == 4:
            date_year = int(date_of_recording.split("/")[1])
            date_month = int(date_of_recording.split("/")[0])
            date_day = 0

    # save all params for updating
    update_dict = {
        "path_file": [image_path, "string"],
        "tma_number": [tma, "integer"],
        "roll": [roll, "string"],
        "frame": [frame, "integer"],
        "view_direction": [view_direction, "string"],
        "altitude": [altitude, "float"],
        "azimuth": [azimuth, "float"],
        "date_of_recording": [date_of_recording, "string"],
        "date_year": [date_year, "integer"],
        "date_month": [date_month, "integer"],
        "date_day": [date_day, "integer"],
        "cam_id": [cam_id, "integer"],
        "location": [location, "string"],
        "x_coords": [x_coords, "float"],
        "y_coords": [y_coords, "float"],
        "point_x": [point_x, "float"],
        "point_y": [point_y, "float"],
        "point": [point, "geom"]
    }

    # create the basis sql-string
    sql_string = "UPDATE images SET "

    # check how many entries we need to update
    nr_updated_entries = 0

    # iterate all fields to update
    for key in update_dict:

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
        p.print_v("No update required for table 'images'", verbose, pbar=pbar)
        return True

    # remove the last trailing comma
    sql_string = sql_string[:-2]

    # add the condition
    sql_string = sql_string + f" WHERE image_id='{image_id}'"

    p.print_v(f"Update table 'images' for '{image_id}'", verbose, pbar=pbar)

    # the real updating
    success = ctd.edit_data_in_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    return success
