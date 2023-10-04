import json
import os
import numpy as np

from tqdm import tqdm

import image_estimations.update_table_images_estimated as utie

import base.connect_to_db as ctd
import base.print_v as p

def correct_focal_lengths(correction_type, min_difference=None, catch=True, verbose=False, pbar=None):

    p.print_v(f"Start: correct_focal_length ({correction_type})", verbose=verbose, pbar=pbar)

    assert correction_type in ["flight_path", "cam_id"]

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the default value for min_difference
    if min_difference is None:
        min_difference = json_data["focal_length_min_difference"]

    if correction_type == "flight_path":
        sql_part = "SUBSTRING(image_id, 3, 7) AS flight_path"
    elif correction_type == "cam_id":
        sql_part = "cam_id"

    # get all focal_length_data
    sql_string = f"SELECT image_id, {sql_part}, " \
                 "focal_length, focal_length_estimated FROM images_extracted"
    data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    # iterate all rows
    for _, row in (pbar := tqdm(data.iterrows())):

        # get all focal lengths with the same flight path
        subset = data[data[correction_type] == row[correction_type]]

        # check for distinct values
        value_counts = subset['focal_length'].value_counts()

        # for these cases we don't need to do anything
        if len(value_counts) <= 1:
            continue

        # Get unique values from value_counts
        unique_vals = value_counts.index.values

        # Check if there are non-zero values in unique_vals
        nonzero_vals = [val for val in unique_vals if int(val * 1000) % 10 != 0]

        if len(nonzero_vals) > 0:
            # Delete values that end with 0.0 at the third decimal from value_counts
            value_counts = value_counts[np.floor(value_counts.index * 1000) % 10 != 0]

        # if we only have one focal_length, it is easy ->
        if value_counts.shape[0] == 1:
            focal_length = value_counts.index[0]
        else:

            print(value_counts)

            most_common_values = value_counts.head(2)

            # get the difference between their values
            difference = most_common_values.iloc[0] - most_common_values.iloc[1]

            # we have a clear winner:
            if difference >= min_difference:
                focal_length = subset['focal_length'].mode()[0]
            else:
                continue

        # no need to update if we already have the right focal length
        if focal_length == row['focal_length']:
            continue

        # correct the focal length
        success = utie.update_table_images_estimated(row['image_id'],
                                                     "focal_length", focal_length,
                                                     catch=catch, verbose=verbose, pbar=pbar)

        if success is False:
            print(row['image_id'])
            break

    p.print_v(f"Finished: correct_focal_length ({correction_type})", verbose=verbose, pbar=pbar)


if __name__ == "__main__":

    correct_focal_lengths("cam_id")