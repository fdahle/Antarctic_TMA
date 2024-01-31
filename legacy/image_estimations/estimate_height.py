import json
import numpy as np
import os

import base.connect_to_db as ctd
import base.print_v as p

debug_show_fid_marks = False


def estimate_height(image_id, min_number_of_images=None, max_std=None,
                    catch=False, verbose=False, pbar=None):
    """
    estimate_height(image_id, min_number_of_images, max_std, catch, verbose, pbar)
        For some images we cannot extract the height. Instead, for these images we
        look at the height of the images of the same flight and then get the
        height based on the other images by taking the most common height
        (and hopefully all heights are identical -> ideal case)
    Args:
        image_id (String): The image-image_id of the image, for which we are estimating height
        min_number_of_images (int): The minimum number of images from the same flight that
            must have fid height
        max_std (float): What is the maximum values that the std of the height can be
            different before we stop estimating
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening
            during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a
            tqdm-progress bar
    Returns:
        height (int): The height as an integer
    """

    p.print_v(f"Start: estimate_height ({image_id})", verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the default values
    if min_number_of_images is None:
        min_number_of_images = json_data["height_estimation_min_nr_images"]

    if max_std is None:
        max_std = json_data["height_estimation_max_std"]

    # get the properties of this image (flight path, etc)
    sql_string = f"SELECT tma_number, view_direction, id_cam FROM images WHERE image_id='{image_id}'"
    data_img_props = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)

    # get the attribute values from panda
    tma_number = data_img_props["tma_number"].iloc[0]
    view_direction = data_img_props["view_direction"].iloc[0]
    cam_id = data_img_props["id_cam"].iloc[0]

    # get the images with the same properties
    sql_string = f"SELECT image_id FROM images WHERE tma_number={tma_number} AND " \
                 f"view_direction='{view_direction}' AND id_cam={cam_id}"
    data_ids = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)

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

    # get all images of the same flight, same camera, same view_direction etc
    sql_string = "SELECT height, height_estimated " \
                 f"FROM images_properties WHERE image_id IN {str_data_ids}"
    height_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    # how many entries can we use
    count = height_data.loc[(height_data['height_estimated'] is False) &
                            (height_data['height'].shape[0] is not np.NaN)].shape[0]

    # if we have too few entries, we don't want to estimate the height
    if count < min_number_of_images:
        return None

    # get the std values
    std_val = height_data.loc[height_data['height_estimated'] is False,
                              'height'].std()

    # if the std is too big, we don't want to estimate the height
    if std_val > max_std:
        return None

    # get the average values
    avg_val = height_data.loc[height_data['height_estimated'] is False,
                              'height'].mean()

    # convert to integer
    avg_val = int(avg_val)

    p.print_v(f"Finished: estimate_height ({image_id})", verbose, pbar=pbar)

    return avg_val
