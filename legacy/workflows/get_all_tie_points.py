import copy
import json
import os

import pandas as pd

from tqdm import tqdm

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.print_v as p
import base.remove_borders as rb
import base.update_failed_ids as uif
import display.display_images

import image_tie_points.find_overlapping_images as foi
import image_tie_points.find_tie_points as ftp1
import image_tie_points.filter_tie_points as ftp2
import image_tie_points.rotate_image_sky as rsk
import image_tie_points.update_table_images_tie_points as utitp

import numpy as np

# smaller image_id is always image_1

matching_mode = "image_id"
use_segmented = False

debug_300_check = True

def get_all_tie_points(image_ids, path_no_tie_points_file=None,
                       path_folder_mask=None, path_folder_segmented=None,
                       overwrite=False, catch=True, verbose=False):
    """
    get_all_tie_points(image_ids, path_folder_mask, path_folder_segmented,
                       overwrite, catch, verbose):
    This function calls all functions that are necessary to find tie-points between the images.
    See the specific function to see what is happening
    Args:
        image_ids (List): A list of image_ids
        path_folder_mask (String): A path to the folder where the masks of the images are located
        path_folder_segmented: A path to the folder where the segmented images are located
        overwrite (Boolean): If true, all values and data will be overwritten with new values,
            if false, existing values and data will not be changed
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of
            the function
    Returns:
         -
    """

    p.print_v(f"Start: get_all_tie_points ({len(image_ids)} images)", verbose=verbose)

    # load the json to get default values
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    json_folder = project_folder + "/image_tie_points"
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if path_no_tie_points_file is None:
        path_no_tie_points_file = json_data["path_file_no_tie_points"]

    if path_folder_mask is None:
        path_folder_mask = json_data["path_folder_masked"]

    if path_folder_segmented is None:
        path_folder_segmented = json_data["path_folder_segmented"]

    # init the FailUpdater
    fup = uif.FailManager("get_all_tie_points")

    # load the file with the no-matches, so that we don't do matching again for images that we know are not working
    data_no_tie_points = pd.read_csv(path_no_tie_points_file, sep=";", header=0)

    for image_id in (pbar := tqdm(image_ids)):

        # get the ids for which we want to find tie-points
        other_ids = foi.find_overlapping_images(image_id, mode=matching_mode,
                                                catch=catch, verbose=verbose, pbar=pbar)

        # check if we have found other ids, if not we cannot find tie-points for this image
        if len(other_ids) == 0:
            p.print_v(f"No other ids could be found for {image_id}", verbose=verbose, pbar=pbar)
            continue

        # check if this image already has all tie-points for other image extracted, then we can continue immediately
        if overwrite is False:
            num_checked_images = 0

            # check all other ids
            for other_id in other_ids:

                # first check if we can find in the database
                sql_string = f"SELECT * FROM images_tie_points WHERE " \
                             f"(image_1_id='{image_id}' AND image_2_id ='{other_id}') OR " \
                             f"(image_1_id='{other_id}' AND image_2_id ='{image_id}')"
                data_table = ctd.get_data_from_db(sql_string,
                                                  catch=catch, verbose=verbose, pbar=pbar)

                # we have a match
                if data_table.shape[0] > 0:
                    num_checked_images = num_checked_images + 1
                    continue

                # check if it is in the no matches
                count = data_no_tie_points.loc[
                    ((data_no_tie_points['image_1_id'] == image_id) & (data_no_tie_points['image_2_id'] == other_id)) |
                    ((data_no_tie_points['image_1_id'] == other_id) & (data_no_tie_points['image_2_id'] == image_id))].shape[0]

                if count > 0:
                    num_checked_images = num_checked_images + 1
                    continue

            # we already checked all images
            if num_checked_images == len(other_ids):
                p.print_v(f"Tie-points for {image_id} are already extracted", verbose, pbar=pbar)
                continue

        # load base image, mask and segmented
        base_img = liff.load_image_from_file(image_id, catch=catch, verbose=verbose, pbar=pbar)
        base_mask = liff.load_image_from_file(image_id, image_path=path_folder_mask,
                                              catch=catch, verbose=verbose, pbar=pbar)
        if use_segmented:
            base_segmented = liff.load_image_from_file(image_id, image_path=path_folder_segmented,
                                                       catch=catch, verbose=verbose, pbar=pbar)

        # remove the borders for all loaded images
        base_img = rb.remove_borders(base_img, image_id=image_id,
                                     catch=catch, verbose=verbose, pbar=pbar)
        base_mask = rb.remove_borders(base_mask, image_id=image_id,
                                      catch=catch, verbose=verbose, pbar=pbar)
        if use_segmented:
            base_segmented = rb.remove_borders(base_segmented, image_id=image_id,
                                               catch=catch, verbose=verbose, pbar=pbar)

        # mask and image need to be equal
        if base_img.shape != base_mask.shape:
            if debug_300_check:
                base_mask = base_mask[300:, :]
                invalid = False
                if base_img.shape != base_mask.shape:
                    invalid = True
            else:
                invalid = True

            if invalid:
                p.print_v(f"The shape of image and mask for {image_id} is not identical", color="red")
                continue

        if use_segmented:
            assert base_img.shape == base_segmented.shape

            # rotate the base image, so that sky is up
            base_img, _, _ = rsk.rotate_image_sky(base_img, base_segmented, image_id,
                                                  catch=catch, verbose=verbose, pbar=pbar)
            base_mask, _, _ = rsk.rotate_image_sky(base_mask, base_segmented, image_id,
                                                   catch=catch, verbose=verbose, pbar=pbar)

        # split up the ids
        base_flight = int(image_id[2:6])
        base_view_direction = image_id[8]
        base_frame = int(image_id[-3:])

        # iterate all other ids
        for other_id in other_ids:

            # load other image, mask and segmented
            other_img = liff.load_image_from_file(other_id,
                                                  catch=catch, verbose=verbose, pbar=pbar)
            other_mask = liff.load_image_from_file(other_id, image_path=path_folder_mask,
                                                   catch=catch, verbose=verbose, pbar=pbar)
            if use_segmented:
                other_segmented = liff.load_image_from_file(other_id, image_path=path_folder_segmented,
                                                            catch=catch, verbose=verbose, pbar=pbar)

            # remove the borders for all loaded images
            other_img = rb.remove_borders(other_img, image_id=other_id,
                                          catch=catch, verbose=verbose, pbar=pbar)
            other_mask = rb.remove_borders(other_mask, image_id=other_id,
                                           catch=catch, verbose=verbose, pbar=pbar)
            if use_segmented:
                other_segmented = rb.remove_borders(other_segmented, image_id=other_id,
                                                    catch=catch, verbose=verbose, pbar=pbar)

            # some checks
            if other_img.shape != other_mask.shape:
                if debug_300_check:
                    other_mask = other_mask[300:,:]
                    invalid = False
                    if other_img.shape != other_mask.shape:
                        invalid = True
                else:
                    invalid = True

                if invalid:
                    p.print_v(f"The shape of image and mask for {image_id} is not identical", color="red")
                    continue

            if use_segmented:
                assert other_img.shape == other_segmented.shape

                # rotate the other image, so that sky is up
                other_img, _, _ = rsk.rotate_image_sky(other_img, other_segmented, other_id,
                                                       catch=catch, verbose=verbose, pbar=pbar)
                other_mask, _, _ = rsk.rotate_image_sky(other_mask, other_segmented, other_id,
                                                        catch=catch, verbose=verbose, pbar=pbar)

            # check which is the smaller image_id, because that is the one that will be image1, the other one will be image 2
            other_flight = int(other_id[2:6])
            other_view_direction = other_id[8]
            other_frame = int(other_id[-3:])

            # if flight is similar, it is easy
            if base_flight == other_flight:
                if base_frame < other_frame:
                    id_1 = image_id
                    image_1 = base_img
                    mask_1 = base_mask
                    id_2 = other_id
                    image_2 = other_img
                    mask_2 = other_mask
                else:
                    id_1 = other_id
                    image_1 = other_img
                    mask_1 = other_mask
                    id_2 = image_id
                    image_2 = base_img
                    mask_2 = base_mask
            elif base_flight < other_flight:
                id_1 = image_id
                image_1 = base_img
                mask_1 = base_mask
                id_2 = other_id
                image_2 = other_img
                mask_2 = other_mask
            else:
                id_1 = other_id
                image_1 = other_img
                mask_1 = other_mask
                id_2 = image_id
                image_2 = base_img
                mask_2 = base_mask

            # we need to check if we don't already have the tie-points
            if overwrite is False:

                # check if we already have tie-points
                sql_string = f"SELECT id FROM images_tie_points WHERE image_1_id='{id_1}' AND image_2_id='{id_2}'"
                data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

                # we already have tie-points
                if data.shape[0] > 0:
                    p.print_v(f"Tie-points already existing for {image_id} and {other_id}", verbose=verbose, pbar=pbar)
                    continue

                # also look in the no matching table
                count = data_no_tie_points.loc[(data_no_tie_points['image_1_id'] == id_1) &
                                               (data_no_tie_points['image_2_id'] == id_2)].shape[0]

                # we have a entry there
                if count > 0:
                    p.print_v(f"There are no tie-points for {image_id} and {other_id}", verbose=verbose, pbar=pbar)
                    continue

            # extract the tie-points for two images
            tie_points, conf = ftp1.find_tie_points(image_1, image_2,
                                                    mask_1=mask_1, mask_2=mask_2,
                                                    catch=catch, verbose=verbose, pbar=pbar)

            # catch extracting errors
            if tie_points is None:
                fup.update_failed_ids(id_1 + "_" + id_2, "find_tie_points")

            # what happens if we didn't find any tie-points
            if tie_points.shape[0] == 0:
                p.print_v(f"No tie-points could be found for {image_id} and {other_id}",
                          verbose, pbar=pbar)

                # add to the dataframe
                row = pd.DataFrame({"image_1_id": image_id,
                                    "image_2_id": other_id}, index=[0])

                data_no_tie_points = pd.concat([data_no_tie_points, row], ignore_index=True)

                continue

            # merge all information in one list
            data = {"tie_points": tie_points, "conf": conf}

            tie_points_filtered, conf_filtered = ftp2.filter_tie_points(tie_points, conf,
                                                                        base_mask, other_mask)

            if tie_points_filtered.shape[0] == 0:
                p.print_v(f"After filtering no tie-points are left for {image_id} and {other_id}",
                          verbose, pbar=pbar)

            # merge all filtered information in one list
            data["tie_points_filtered"] = tie_points_filtered
            data["conf_filtered"] = conf_filtered

            # add the filtered tie-points to the table
            success = utitp.update_table_images_tie_points(id_1, id_2, data, overwrite=overwrite,
                                                           catch=catch, verbose=verbose, pbar=pbar)

            if success is False:
                fup.update_failed_ids(id_1 + "_" + id_2, "update_table_images_tie_points:filtered")

        # drop data to csv
        data_no_tie_points.to_csv(path_no_tie_points_file, index=False, sep=";", header=True)


if __name__ == "__main__":

    img_ids = ['CA184632V0327', 'CA184632V0330', 'CA184632V0334', 'CA184632V0335', 'CA184632V0342', 'CA184632V0318',
     'CA184632V0323', 'CA184632V0324', 'CA184632V0321', 'CA184632V0320', 'CA184632V0326', 'CA184632V0319',
     'CA184632V0317', 'CA184632V0328', 'CA184632V0337', 'CA184632V0332', 'CA184632V0344', 'CA184632V0316',
     'CA184632V0331', 'CA184632V0329', 'CA184632V0343', 'CA184632V0322', 'CA184632V0336', 'CA184632V0340',
     'CA184632V0325', 'CA184632V0338', 'CA184632V0341', 'CA184632V0333', 'CA184632V0345', 'CA184632V0339']

    get_all_tie_points(img_ids, overwrite=False, catch=False, verbose=True)
