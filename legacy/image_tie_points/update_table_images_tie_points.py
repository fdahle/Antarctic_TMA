import numpy as np

import base.connect_to_db as ctd
import base.print_v as p


def update_table_images_tie_points(image_id_1, image_id_2, data,
                                   overwrite=True, catch=True, verbose=False, pbar=None):
    """
    update_table_images_tie_points(image_id_1, image_id_2, data, overwrite, catch, verbose, pbar):
    update the table images_tie_points with the tie-points between two images. Note that here no
    'sql-update' is happening, but an 'insert into'
    Args:
        image_id_1 (str): The image-image_id of the first (left) image where we found tie-points
        image_id_2 (str): The image-image_id of the second (right) image where we found tie-points
        data (pandas): The tie-points and their quality as a dataframe
        overwrite (Boolean): If this is true, we overwrite regardless of existing data
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        success (Boolean): If this is false, something went wrong during execution
    """

    p.print_v(f"Start: update_table_images_tie_points ({image_id_1}, {image_id_2})",
              verbose=verbose, pbar=pbar)

    # get the values from our data
    extraction_method = "LightGlue"
    x1 = data["tie_points"][:, 0]
    y1 = data["tie_points"][:, 1]
    x2 = data["tie_points"][:, 2]
    y2 = data["tie_points"][:, 3]
    quality = np.around(np.array(data["conf"]), 3)

    if data["tie_points_filtered"].shape[0] > 0:
        x1_filtered = data["tie_points_filtered"][:, 0]
        y1_filtered = data["tie_points_filtered"][:, 1]
        x2_filtered = data["tie_points_filtered"][:, 2]
        y2_filtered = data["tie_points_filtered"][:, 3]
        quality_filtered = np.around(np.array(data["conf_filtered"]), 3)

    # convert to the format we want to have it in our sql-query
    tie_points_image_1 = ""
    tie_points_image_2 = ""
    quality_str = ""
    for i in range(data["tie_points"].shape[0]):
        tie_points_image_1 = tie_points_image_1 + str(x1[i]) + "," + str(y1[i]) + ";"
        tie_points_image_2 = tie_points_image_2 + str(x2[i]) + "," + str(y2[i]) + ";"
        quality_str = quality_str + str(quality[i]) + ";"
    tie_points_image_1 = tie_points_image_1[:-1]
    tie_points_image_2 = tie_points_image_2[:-1]
    quality_str = quality_str[:-1]

    # convert to the format we want to have it in our sql-query for filtered
    tie_points_image_1_filtered = ""
    tie_points_image_2_filtered = ""
    quality_filtered_str = ""
    if data["tie_points_filtered"].shape[0] > 0:
        for i in range(data["tie_points_filtered"].shape[0]):
            tie_points_image_1_filtered = tie_points_image_1_filtered + \
                                          str(x1_filtered[i]) + "," + str(y1_filtered[i]) + ";"  # noqa
            tie_points_image_2_filtered = tie_points_image_2_filtered + \
                                          str(x2_filtered[i]) + "," + str(y2_filtered[i]) + ";"  # noqa
            quality_filtered_str = quality_filtered_str + str(quality_filtered[i]) + ";"  # noqa
        tie_points_image_1_filtered = tie_points_image_1_filtered[:-1]
        tie_points_image_2_filtered = tie_points_image_2_filtered[:-1]
        quality_filtered_str = quality_filtered_str[:-1]

    # calculate some values
    number_tie_points = data["tie_points"].shape[0]
    number_tie_points_filtered = data["tie_points_filtered"].shape[0]
    avg_quality = round(quality.mean(), 3)
    avg_quality_filtered = 0
    if data["tie_points_filtered"].shape[0] > 0:
        avg_quality_filtered = round(quality_filtered.mean(), 3)

    # first check if the data is already in
    sql_string = f"SELECT * FROM images_tie_points WHERE " \
                 f"image_1_id='{image_id_1}' and image_2_id='{image_id_2}'"
    table_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    # case 1: data is not in there -> Create first
    if table_data.shape[0] == 0:
        sql_string = "INSERT INTO images_tie_points (image_1_id, image_2_id) VALUES " \
                     f"('{image_id_1}', '{image_id_2}')"
        success = ctd.add_data_to_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

        if success is False:
            return False

    # if we don't want to overwrite we need to cancel this operation
    if table_data.shape[0] == 1 and overwrite is False:
        return True

    # sql-string for updating the values
    sql_string = "UPDATE images_tie_points SET " \
                 f"extraction_method='{extraction_method}', " \
                 f"number_tie_points={number_tie_points}, " \
                 f"number_tie_points_filtered={number_tie_points_filtered}, " \
                 f"avg_quality={avg_quality}, " \
                 f"avg_quality_filtered={avg_quality_filtered}, " \
                 f"tie_points_image_1='{tie_points_image_1}', " \
                 f"tie_points_image_2='{tie_points_image_2}', " \
                 f"quality='{quality_str}', " \
                 f"tie_points_image_1_filtered='{tie_points_image_1_filtered}', " \
                 f"tie_points_image_2_filtered='{tie_points_image_2_filtered}', " \
                 f"quality_filtered='{quality_filtered_str}' " \
                 f"WHERE image_1_id='{image_id_1}' and image_2_id='{image_id_2}'"

    success = ctd.edit_data_in_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    if success:
        p.print_v(f"Start: update_table_images_tie_points ({image_id_1}, {image_id_2})",
                  verbose=verbose, pbar=pbar)
    else:
        p.print_v(f"Failed: update_table_images_tie_points ({image_id_1}, {image_id_2})",
                  verbose=verbose, pbar=pbar)

    return success
