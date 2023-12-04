import json
import os
import numpy as np

from PIL import Image
from tqdm import tqdm

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.print_v as p
import base.remove_borders as rb
import base.update_failed_ids as ufi

import image_masking.mask_border as mb
import image_masking.mask_segmented as ms
import image_masking.mask_text as mt

debug_print_all = True

# 0 means masked, 1 means not masked

def mask_all_images(image_ids,
                    mask_folder=None, segmented_folder=None,
                    overwrite=False, catch=True, verbose=False):
    """
    mask_all_images(
    Args:
        image_ids:
        mask_folder:
        segmented_folder:
        overwrite:
        catch:
        verbose:

    Returns:

    """

    p.print_v(f"Start: mask_all_images ({len(image_ids)} images)", verbose=verbose)

    # load the json to get default values
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    json_folder = project_folder + "/image_masking"
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if mask_folder is None:
        mask_folder = json_data["path_folder_mask"]

    if segmented_folder is None:
        segmented_folder = json_data["path_folder_segmented"]

    # initialize the failed image_id manager
    fail_manager = ufi.FailManager("mask_all_images")

    # iterate all images
    for image_id in (pbar := tqdm(image_ids)):

        if debug_print_all:
            pbar=None

        p.print_v(f"Runtime: mask_all_images: {image_id}", verbose=verbose, pbar=pbar)

        # check if the image is already existing
        path_mask_file = mask_folder + "/" + image_id + ".tif"

        if overwrite is False and os.path.isfile(path_mask_file):
            p.print_v(f"There is already a mask existing for {image_id}", verbose, pbar=pbar)
            continue

        # load the image from file
        img = liff.load_image_from_file(image_id,
                                        catch=catch, verbose=verbose, pbar=pbar)

        if img is None:
            fail_manager.update_failed_id(image_id, "load_image_from_file")
            continue

        # create a mask with the same size as the image
        mask = np.ones_like(img)

        # load borders
        _, border_bounds = rb.remove_borders(img, image_id, cut_method="auto",
                                             return_edge_dims=True,
                                             catch=catch, verbose=verbose, pbar=pbar)

        if border_bounds is None:
            fail_manager.update_failed_id(image_id, "remove_borders")
            continue

        mask = mb.mask_borders(mask, border_bounds, catch=catch, verbose=verbose, pbar=pbar)

        if mask is None:
            fail_manager.update_failed_id(image_id, "mask_borders")
            continue

        # load segmented
        segmented = liff.load_image_from_file(image_id, image_path=segmented_folder,
                                              catch=catch, verbose=verbose, pbar=pbar)

        if segmented is None:
            fail_manager.update_failed_id(image_id, "load_image_from_file:segmented")
            continue

        # add segmented to mask
        mask = ms.mask_segmented(mask, segmented,
                                 catch=catch, verbose=verbose, pbar=pbar)

        if mask is None:
            fail_manager.update_failed_id(image_id, "mask_segmented")
            continue

        # load the text positions
        sql_string = f"SELECT text_bbox FROM images_extracted WHERE " \
                     f"image_id='{image_id}'"
        text_pos = ctd.get_data_from_db(sql_string,
                                        catch=catch, verbose=verbose, pbar=pbar)
        if len(text_pos) > 0:
            text_pos = text_pos.iloc[0]

            if text_pos["text_bbox"] is None:
                fail_manager.update_failed_id(image_id, "get_data_from_db:images_extracted")
                continue

            # add text to the mask
            mask = mt.mask_text(mask, text_pos["text_bbox"],
                                catch=catch, verbose=verbose, pbar=pbar)

            if mask is None:
                fail_manager.update_failed_id(image_id, "get_data_from_db:mask_text")
                continue

        print(f"Image saved at {path_mask_file}")

        # save the image
        im = Image.fromarray(mask)
        im.save(path_mask_file)

    # save the failed ids
    fail_manager.save_csv()

    p.print_v(f"Finished: mask_all_images ({len(image_ids)}) images", verbose=verbose)


if __name__ == "__main__":
    img_ids = ['CA179231L0038']

    seg_folder = "/data_1/ATM/data_1/aerial/TMA/segmented/supervised"

    mask_all_images(img_ids, segmented_folder=seg_folder,
                    overwrite=True, catch=False, verbose=True)
