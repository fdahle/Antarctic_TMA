import copy
import cv2
import json
import numpy as np
import os
import mahotas as mht

import base.connect_to_db as ctd
import base.print_v as p

import display.display_images as di

debug_show_sidebars = False


def correct_image_orientation_sidebar(img, image_id, extract_width=None, cut_off=None,
                                      force=False, overwrite=True, catch=True, verbose=False, pbar=None):

    """
    correct_image_orientation_sidebar(input_img, extract_width, cut_off, verbose, catch):
    Many images were upside down while scanning so that the information bar for some images is
    on the right side (it should be on the left side). It can be seen for example with the
    clock (because the numbers are upside down). This function tries to rotate the images to
    the right way: From every side a subset is taken and these images are checked for
    homogeneity (after blurring). The more homogenous part is the one without the
    information bar (and should be on the right side).
    Args:
        img (np-arr): The image we want to check for sidebar-position
        image_id (String): The image_id of the image we are checking
        extract_width (Int, 300): How many pixels of the left/right side are we checking
        cut_off (Int, 3000): Remove this amount of pixels from the top and the button to
            speed up the process
        force (Boolean): If true, we will rotate the image without even checking it
        overwrite (Boolean): If true and correction is already checked we don't need to do the whole operation
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        img (Boolean): The rotated (or non rotated) image
    """

    p.print_v(f"Start: correct_image_orientation_sidebar ({image_id})", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # we need the image (of course) but also the image_id
    assert img is not None, "No image could be found"
    assert image_id is not None, "Image-image_id is missing"

    # get the orientation information from the db
    sql_string = f"SELECT path_file, rotation_sidebar_corrected FROM images WHERE image_id='{image_id}'"
    data_image = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    # check if we need to correct the image rotation
    bool_rot = data_image["rotation_sidebar_corrected"].iloc[0]
    if overwrite is False and str(bool_rot) == 'True':
        p.print_v(f"{image_id} already is rotated", verbose, "green", pbar)
        return img, True

    if extract_width is None:
        extract_width = json_data["correct_image_rotation_sidebar_width"]

    if cut_off is None:
        cut_off = json_data["correct_image_rotation_sidebar_cutoff"]

    # hardcopy image
    img = copy.deepcopy(img)

    # for east and west
    cut_off_e = extract_width
    cut_off_w = extract_width

    # init the variable for later
    must_rotate = False

    try:

        # extract subset
        extracted_e = img[cut_off:img.shape[0] - cut_off, :cut_off_w]
        extracted_w = img[cut_off:img.shape[0] - cut_off, img.shape[1]-cut_off_e:img.shape[1]]

        # blur image
        blurred_e = cv2.GaussianBlur(extracted_e, (9, 9), 0)  # noqa
        blurred_w = cv2.GaussianBlur(extracted_w, (9, 9), 0)  # noqa

        # make binary
        _, blurred_e = cv2.threshold(blurred_e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, blurred_w = cv2.threshold(blurred_w, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # calculate homogeneity
        homog_e = mht.features.haralick(blurred_e).mean(0)[1]
        homog_w = mht.features.haralick(blurred_w).mean(0)[1]

        if debug_show_sidebars:
            di.display_images([img, blurred_w, blurred_e],
                              title=str(round(homog_w, 5)) + " " + str(round(homog_e, 5)))

        # check if we must rotate
        if homog_w > homog_e:
            must_rotate = True

        # calculate difference
        diff = abs(homog_w - homog_e)

        # difference is too small and rotation cannot be saved for sure
        if diff < 0.1 and force is False:
            p.print_v("Difference too small to apply rotation with confidence.", verbose, color="yellow", pbar=pbar)
            return None, False

        # image must be rotated (because of the check or because we force it)
        if must_rotate or force:

            # rotate image
            img = np.rot90(img, 2)

        sql_string = f"UPDATE images SET rotation_sidebar_corrected=True WHERE image_id='{image_id}'"
        success = ctd.edit_data_in_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

        # save the new rotated
        cv2.imwrite(data_image['path_file'].iloc[0], img)

        p.print_v(f"Finished: correct_image_orientation_sidebar ({image_id})", verbose=verbose, pbar=pbar)

        return img, success

    except (Exception,) as e:

        if catch:
            p.print_v(f"Failed: correct_image_orientation_sidebar ({image_id})", verbose, color="red", pbar=pbar)
            return None, False
        else:
            raise e


if __name__ == "__main__":

    img_ids = ['CA182431L0007', 'CA182331L0073', 'CA182131L0093', 'CA181633R0027']

    for img_id in img_ids:

        import base.load_image_from_file as liff
        image = liff.load_image_from_file(img_id, catch=False)

        correct_image_orientation_sidebar(image, img_id, force=True, catch=True, verbose=True)
