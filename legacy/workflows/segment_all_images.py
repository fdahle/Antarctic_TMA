import copy
import cv2
import os
import json
import numpy as np

from tqdm import tqdm

import display.display_images
import image_segmentation.get_image_rotation_sky as girs
import image_segmentation.improve_segmentation as ise
import image_segmentation.segment_image as si
import image_segmentation.update_table_segmented as uts

import base.load_image_from_file as liff
import base.print_v as p
import base.remove_borders as rb
import base.update_failed_ids as ufi

debug_remove_borders = True
debug_improve_segmentation = True


# the segmentation has the following classes:
# 1: ice, 2: snow, 3: rocks, 4: water, 5: clouds, 6:sky, 7: unknown


def segment_all_images(image_ids, path_segmentation_folder=None,
                       overwrite=False, catch=True, verbose=False):
    """

    Args:
        image_ids:
        path_segmentation_folder:
        overwrite:
        catch:
        verbose:

    Returns:

    """

    p.print_v(f"Start: segment_all_images ({len(image_ids)} images)", verbose=verbose)

    # load the json to get default values
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    json_folder = project_folder + "/image_segmentation"
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the segmentation folder
    if path_segmentation_folder is None:
        path_segmentation_folder = json_data["path_folder_segmented"]

    # initialize the failed image_id manager
    fail_manager = ufi.FailManager("segment_all_images")

    # iterate all image_ids
    for image_id in (pbar := tqdm(image_ids)):

        # get the path where the segmented file would be saved
        path_segmented_file = path_segmentation_folder + "/" + image_id + ".tif"

        # check if we already have an image and possible cut short due to overwrite
        if overwrite is False and os.path.exists(path_segmented_file):
            p.print_v(f"Runtime: {image_id} is already segmented", verbose=verbose, pbar=pbar)
            continue

        # load the image
        image = liff.load_image_from_file(image_id, catch=catch, verbose=verbose, pbar=pbar)

        # check if something went wrong with loading the image
        if image is None:
            fail_manager.update_failed_id(image_id, "load_image_from_file")
            continue

        # save image with original shape
        orig_image = copy.deepcopy(image)

        # remove the border
        if debug_remove_borders:
            image, borders = rb.remove_borders(image, image_id, return_edge_dims=True,
                                               cut_method="database",
                                               catch=catch, verbose=verbose, pbar=pbar)

            if image is None:
                fail_manager.update_failed_id(image_id, "remove_borders")
                continue

        else:
            borders = None

        # get the segmented version of the image
        segmented, probabilities, highest_prob, model_name = si.segment_image(image, image_id,
                                                                              catch=catch, verbose=verbose, pbar=pbar)

        # check if something went wrong with segmenting
        if segmented is None:
            fail_manager.update_failed_id(image_id, "segment_image")
            continue

        if debug_improve_segmentation:
            # check if the images are rotated correctly (sky)
            correct_rotated = girs.get_image_rotation_sky(segmented, image_id,
                                                          catch=catch, verbose=verbose, pbar=pbar)

            if correct_rotated is None:
                fail_manager.update_failed_id(image_id, "correct_rotated")
                continue

            # rotate image so that sky is top
            if correct_rotated is False:
                segmented = segmented[::-1, ::-1]
                probabilities = probabilities[::-1, ::-1]

            # improve the segmented image
            segmented_improved = ise.improve_segmentation(image_id, segmented, probabilities,
                                                          catch=catch, verbose=verbose, pbar=pbar)

            # check if something went wrong with improving segmentation
            if segmented_improved is None:
                fail_manager.update_failed_id(image_id, "segmented_improved")
                continue

            # rotate image back
            if correct_rotated is False:
                segmented_improved = segmented_improved[::-1, ::-1]

        # no improvement wished
        else:
            segmented_improved = segmented

        # add the borders back to the image
        if debug_remove_borders:
            segmented_improved = np.pad(segmented_improved,
                                        pad_width=((borders[2], orig_image.shape[0] - borders[3]),
                                                   (borders[0], orig_image.shape[1] - borders[1])),
                                        mode='constant', constant_values=7)

        # some additional information for the updating of the table
        update_data = {
            "labelled_by": "unet",
            "model_name": model_name
        }

        # update the table
        success = uts.update_table_segmented(image_id, segmented_improved, update_data,
                                             overwrite=overwrite, catch=catch, verbose=verbose, pbar=pbar)

        if success is None:
            fail_manager.update_failed_id(image_id, "update_table_segmented")
            continue

        # save the segmented image
        cv2.imwrite(path_segmented_file, segmented_improved)

    # save the failed ids
    fail_manager.save_csv()

    p.print_v(f"Finished: segment_all_images ({len(image_ids)}) images", verbose=verbose)


if __name__ == "__main__":

    file_path = "/data_1/ATM/data_1/playground/img_ids_peninsula.txt"
    with open(file_path, "r") as file:
        # Read the entire contents of the file
        text = file.read()
        text = text.strip("\n")

    img_ids = text.split(";")

    segment_all_images(img_ids, overwrite=False, catch=True, verbose=True)
