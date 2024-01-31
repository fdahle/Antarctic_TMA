import json
import numpy as np
import os

import base.connect_to_db as ctd
import base.print_v as p


def estimate_subsets(image_id, min_number_of_images=None, max_std=None,
                     catch=True, verbose=False, pbar=None):
    """
    estimate_subsets(image_id, min_number_of_images, max_std, catch, verbose, pbar):
    For some images we cannot extract subsets. Instead, for these images we look at the subsets of the images of
    the same flight and then estimate the subset based on the other images by taking the average x- and y-position.
    Args:
        image_id (String): The image-image_id of the image, for which we are estimating subsets
        min_number_of_images (int): The minimum number of images from the same flight that must have subsets
        max_std (Float): The maximum standard deviation how much the subset positions can deviate from each other
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        subset_estimations (dict): A dict containing the estimated subsets in the
            format {subset_1_x: xx, subset_1_y: xx}
    """

    p.print_v(f"Start: estimate_subsets ({image_id})", verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the default values
    if min_number_of_images is None:
        min_number_of_images = json_data["subset_estimation_min_nr_images"]

    if max_std is None:
        max_std = json_data["subset_estimation_max_std"]

    # get the properties of this image (flight path, etc)
    sql_string = f"SELECT tma_number, view_direction, id_cam FROM images WHERE image_id='{image_id}'"
    data_img_props = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    # get the attribute values from panda
    tma_number = data_img_props["tma_number"].iloc[0]
    view_direction = data_img_props["view_direction"].iloc[0]

    # get the images with the same properties
    sql_string = f"SELECT image_id FROM images WHERE tma_number={tma_number} AND " \
                 f"view_direction='{view_direction}'"
    data_ids = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    # convert to list and flatten
    data_ids = data_ids.values.tolist()
    data_ids = [item for sublist in data_ids for item in sublist]

    # remove the image_id from the image we want to extract information from
    data_ids.remove(image_id)

    # check if we still have data
    if len(data_ids) == 0:
        p.print_v(f"No other data could be found for '{image_id}'", verbose, pbar=pbar)
        return None

    # convert list to a string
    str_data_ids = "('" + "', '".join(data_ids) + "')"

    # get all entries from the same flight and the same viewing direction
    sql_string = "SELECT image_id, subset_n_x, subset_n_y, subset_e_x, subset_e_y, " \
                 "subset_s_x, subset_s_y, subset_w_x, subset_w_y, " \
                 "subset_n_estimated, subset_e_estimated, " \
                 "subset_s_estimated, subset_w_estimated " \
                 f"FROM images_properties WHERE image_id IN {str_data_ids}"
    subset_data = ctd.get_data_from_db(sql_string)

    # here we will store the results
    subset_estimations = {}

    # estimate the values for all direction
    for direction in ["n", "e", "s", "w"]:

        # init the dict value -> later we fill it with the real values
        subset_estimations[direction + "_x"] = None
        subset_estimations[direction + "_y"] = None

        # count the number of non Nan values (x and y should be similar)
        x_count = subset_data.loc[(subset_data[f'subset_{direction}_estimated'] is False) &
                                  (subset_data[f'subset_{direction}_x'] is not np.NaN)].shape[0]
        y_count = subset_data.loc[(subset_data[f'subset_{direction}_estimated'] is False) &
                                  (subset_data[f'subset_{direction}_y'] is not np.NaN)].shape[0]

        if x_count < min_number_of_images or y_count < min_number_of_images:
            continue

        # get the std values
        x_std = subset_data.loc[subset_data[f'subset_{direction}_estimated'] is False,
                                f'subset_{direction}_x'].std()
        y_std = subset_data.loc[subset_data[f'subset_{direction}_estimated'] is False,
                                f'subset_{direction}_y'].std()

        if x_std > max_std or y_std > max_std:
            continue

        # get the average values
        x_val = subset_data.loc[subset_data[f'subset_{direction}_estimated'] is False,
                                f'subset_{direction}_x'].mean()
        y_val = subset_data.loc[subset_data[f'subset_{direction}_estimated'] is False,
                                f'subset_{direction}_y'].mean()

        # convert to integer
        x_val = int(x_val)
        y_val = int(y_val)

        # save in our result dict
        subset_estimations[direction + "_x"] = x_val
        subset_estimations[direction + "_y"] = y_val

    p.print_v(f"Finished: estimate_subsets ({image_id})", verbose, pbar=pbar)

    return subset_estimations
