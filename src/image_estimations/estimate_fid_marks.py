import json
import numpy as np
import os

import base.connect_to_db as ctd
import base.print_v as p

debug_show_fid_marks = False


def estimate_fid_marks(image_id, min_number_of_images=None, max_std=None,
                       catch=True, verbose=False, pbar=None):
    """
    estimate_fid_marks(image_id, min_number_of_images, max_std, catch, verbose, pbar)
    For some images we cannot extract fid-marks. Instead, for these images we look at the fid-marks of the images of
    the same flight and then estimate the fid-marks based on the other images by taking the average x- and y-position.
    Args:
        image_id (String): The image-image_id of the image, for which we are estimating fid marks
        min_number_of_images (int): The minimum number of images from the same flight that must have fid marks
        max_std (Float): The maximum standard deviation how much the fid marks can deviate from each other
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        fid_estimations (dict): A dict containing the estimated fid marks in the
            format {fid_mark_1_x: xx, fid_mark_1_y: xx}
    """

    p.print_v(f"Start: estimate_fid_marks ({image_id})", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the default value for min_number_of_images
    if min_number_of_images is None:
        min_number_of_images = json_data["fid_estimation_min_nr_images"]

    # get the default value for max_std
    if max_std is None:
        max_std = json_data["fid_estimation_max_std"]

    # get the properties of this image (flight path, etc)
    sql_string = f"SELECT tma_number, view_direction, id_cam FROM images WHERE image_id='{image_id}'"
    data_img_props = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    # get the attribute values from panda
    tma_number = data_img_props["tma_number"].iloc[0]
    view_direction = data_img_props["view_direction"].iloc[0]
    cam_id = data_img_props["id_cam"].iloc[0]

    # get the images with the same properties
    sql_string = f"SELECT image_id FROM images WHERE tma_number={tma_number} AND " \
                 f"view_direction='{view_direction}' AND id_cam={cam_id}"
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

    # get all images of the same flight, same camera, same view_direction etc
    sql_string = "SELECT image_id, fid_mark_1_x , fid_mark_1_y, fid_mark_2_x, fid_mark_2_y, " \
                 "fid_mark_3_x, fid_mark_3_y, fid_mark_4_x, fid_mark_4_y, " \
                 "fid_mark_5_x, fid_mark_5_y, fid_mark_6_x, fid_mark_6_y, " \
                 "fid_mark_7_x, fid_mark_7_y, fid_mark_8_x, fid_mark_8_y, " \
                 "fid_mark_1_estimated, fid_mark_2_estimated, fid_mark_3_estimated, fid_mark_4_estimated, " \
                 "fid_mark_5_estimated, fid_mark_6_estimated, fid_mark_7_estimated, fid_mark_8_estimated " \
                 f"FROM images_properties WHERE image_id IN {str_data_ids}"
    fid_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    # here we will store the results
    fid_estimations = {}

    # now we can estimate a fid mark for every one of the fid marks
    for fid_mrk in range(1, 9):

        # init the dict value -> later we fill it with the real values
        fid_estimations[str(fid_mrk) + "_x"] = None
        fid_estimations[str(fid_mrk) + "_y"] = None

        # how many entries can we use
        x_count = fid_data.loc[(fid_data[f'fid_mark_{fid_mrk}_estimated'] is False) &
                               (fid_data[f'fid_mark_{fid_mrk}_x'] is not np.NaN)].shape[0]
        y_count = fid_data.loc[(fid_data[f'fid_mark_{fid_mrk}_estimated'] is False) &
                               (fid_data[f'fid_mark_{fid_mrk}_y'] is not np.NaN)].shape[0]

        if x_count < min_number_of_images or y_count < min_number_of_images:
            continue

        # get the std values
        x_std = fid_data.loc[fid_data[f'fid_mark_{fid_mrk}_estimated'] is False,
                             f'fid_mark_{fid_mrk}_x'].std()
        y_std = fid_data.loc[fid_data[f'fid_mark_{fid_mrk}_estimated'] is False,
                             f'fid_mark_{fid_mrk}_y'].std()

        if x_std > max_std or y_std > max_std:
            continue

        # get the average values
        x_val = fid_data.loc[fid_data[f'fid_mark_{fid_mrk}_estimated'] is False,
                             f'fid_mark_{fid_mrk}_x'].mean()
        y_val = fid_data.loc[fid_data[f'fid_mark_{fid_mrk}_estimated'] is False,
                             f'fid_mark_{fid_mrk}_y'].mean()

        # convert to integer
        x_val = int(x_val)
        y_val = int(y_val)

        # save in our result dict
        fid_estimations[str(fid_mrk) + "_x"] = x_val
        fid_estimations[str(fid_mrk) + "_y"] = y_val

    p.print_v(f"Finished: estimate_fid_marks ({image_id})", verbose=verbose, pbar=pbar)

    return fid_estimations
