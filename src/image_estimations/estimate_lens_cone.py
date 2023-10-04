import json
import os

import base.connect_to_db as ctd
import base.print_v as p


def estimate_lens_cone(image_id, min_number_of_images=None,
                       min_difference=None,
                       catch=True, verbose=False, pbar=None):
    """
    estimate_lens_cone(image_id, min_number_of_images, max_std, catch, verbose, pbar)
        For some images we cannot extract of find the cam-image_id. Instead, for these images we
        look at the cam-image_id of the images of the same flight and then get the
        cam-image_id based on the other images by taking the most common cam-image_id
        (and hopefully all cam-ids are identical -> ideal case)
    Args:
        image_id (String): The image-image_id of the image, for which we are estimating lens cones
        min_number_of_images (int): The minimum number of images from the same flight that
            must have lens cones
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the
            function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        lens_cone (float): The lens cone as a float number
    """

    p.print_v(f"Start: estimate_lens_cone ({image_id})", verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the default value for min_number_of_images
    if min_number_of_images is None:
        min_number_of_images = json_data["lens_cone_min_nr_images"]

    # get the default value for min_difference
    if min_difference is None:
        min_difference = json_data["lens_cone_min_difference"]

    # get the properties of this image (flight path, etc)
    sql_string = f"SELECT tma_number, view_direction, cam_id FROM images WHERE image_id='{image_id}'"
    data_images = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)

    # if we don't have any information on the images we cannot estimate the lens cone
    if data_images is None:
        return None

    # get the attribute values from panda
    tma_number = data_images["tma_number"].iloc[0]
    view_direction = data_images["view_direction"].iloc[0]
    cam_id = data_images["cam_id"].iloc[0]

    # get the images with the same properties
    sql_string = f"SELECT image_id FROM images WHERE tma_number={tma_number} AND " \
                 f"view_direction='{view_direction}' AND cam_id={cam_id}"
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

    # get the lens cone data
    sql_string = "SELECT lens_cone, lens_cone_estimated " \
                 f"FROM images_extracted WHERE image_id IN {str_data_ids}"
    lens_cone_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    # count how many lens cones we have that are good
    count = lens_cone_data[(lens_cone_data['lens_cone_estimated'] == False) &  # noqa
                           lens_cone_data['lens_cone'].notnull()].shape[0]

    # if we have too few entries, we don't want to estimate the lens cone
    if count < min_number_of_images:
        return None

    # check for distinct values
    unique_values = lens_cone_data['lens_cone'].value_counts()

    # if we only have one lens cone, it is easy ->
    if unique_values.shape[0] == 1:
        lens_cone = unique_values.index[0]
    else:

        most_common_values = unique_values.head(2)

        # get the difference between their values
        difference = most_common_values.iloc[0] - most_common_values.iloc[1]

        # we have a clear winner:
        if difference >= min_difference:
            lens_cone = lens_cone_data['lens_cone'].mode()[0]
        else:
            return None

    p.print_v(f"Finished: estimate_lens_cone({image_id})", verbose, pbar=pbar)

    return lens_cone


if __name__ == "__main__":
    _img_id = "CA135832V0091"

    _lens_cone = estimate_lens_cone(_img_id, catch=False, verbose=True)
    print(_lens_cone)
