import json
import os
from tqdm import tqdm

import image_estimations.update_table_images_estimated as utie
import base.connect_to_db as ctd


def correct_cam_ids(min_difference=None, catch=True, verbose=False, pbar=None):
    """
    correct_cam_ids(min_difference, catch, verbose, pbar):

    Args:
        min_difference:
        catch:
        verbose:
        pbar:

    Returns:

    """

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the default value for min_difference
    if min_difference is None:
        min_difference = json_data["cam_id_min_difference"]

    # get all data for cam _id
    sql_string = f"SELECT image_id, SUBSTRING(image_id, 3, 7) AS flight_path, " \
                 "cam_id, cam_id_estimated FROM images_extracted"
    data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    # iterate all rows
    for _, row in (pbar := tqdm(data.iterrows())):

        # get all focal lengths with the same flight path
        subset = data[data["flight_path"] == row["flight_path"]]

        # check for distinct values
        value_counts = subset['cam_id'].value_counts()

        # for these cases we don't need to do anything
        if len(value_counts) <= 1:
            continue

        # if we only have one focal_length, it is easy ->
        if value_counts.shape[0] == 1:
            cam_id = value_counts.index[0]
        else:

            most_common_values = value_counts.head(2)

            # get the difference between their values
            difference = most_common_values.iloc[0] - most_common_values.iloc[1]

            # we have a clear winner:
            if difference >= min_difference:
                cam_id = subset['cam_id'].mode()[0]
            else:
                continue

        # no need to update if we already have the right focal length
        if cam_id == row['cam_id']:
            continue

        # correct the focal length
        success = utie.update_table_images_estimated(row['image_id'],
                                                     "cam_id", cam_id,
                                                     catch=catch, verbose=verbose, pbar=pbar)

        if success is False:
            print(row['image_id'])
            break


if __name__ == "__main__":

    correct_cam_ids()
