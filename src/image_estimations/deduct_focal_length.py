import os
import json

import base.connect_to_db as ctd


def deduct_focal_length(image_id, min_number_of_images=None, min_difference=None,
                        catch=True, verbose=False, pbar=None):

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the default value for min_number_of_images
    if min_number_of_images is None:
        min_number_of_images = json_data["focal_length_min_nr_images"]

    # get the default value for min_difference
    if min_difference is None:
        min_difference = json_data["focal_length_min_difference"]

    # get the current focal length & cam-image_id for this image
    sql_string = f"SELECT focal_length, cam_id FROM images_extracted WHERE image_id='{image_id}'"
    data = ctd.get_data_from_db(sql_string)
    focal_length = data.iloc[0]['focal_length']
    cam_id = data.iloc[0]['cam_id']

    # select all focal-length of this cam-image_id
    sql_string = f"SELECT focal_length from images_extracted WHERE cam_id='{cam_id}'"
    focal_lengths = ctd.get_data_from_db(sql_string)

    # if we have too few entries, we don't want to estimate the focal length
    if focal_lengths.shape[0] < min_number_of_images:
        return None

    # check for distinct values
    value_counts = focal_lengths['focal_length'].value_counts()

    # if we only have one focal_length, it is easy ->
    if value_counts.shape[0] == 1:
        focal_length = value_counts.index[0]
    else:

        most_common_values = value_counts.head(2)

        # something went wrong and so we return no focal length
        if len(value_counts) == 0:
            return None

        # get the difference between their values
        difference = most_common_values.iloc[0] - most_common_values.iloc[1]

        # we have a clear winner:
        if difference >= min_difference:
            focal_length = focal_lengths['focal_length'].mode()[0]
        else:
            return None

    return focal_length


if __name__ == "__main__":

    img_id = "CA213831L0152"

    deduct_focal_length(img_id)