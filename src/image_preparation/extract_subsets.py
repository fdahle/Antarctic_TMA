import copy
import cv2
import dlib
import json
import os
import numpy as np

import base.connect_to_db as ctd
import base.print_v as p

import display.display_images as di

# debug_params
debug_show_crops = False  # the initial crops
debug_show_subsets_crop = False  # show the subsets in the crop
debug_show_subsets_total = False  # show the subsets in the complete image


def extract_subsets(image, image_id, detector_path=None, model_name=None, fid_type=None,
                    crop_factor=None, binarize_subset=False,
                    overwrite=True, catch=True, verbose=False, pbar=None):
    """
    extract_subsets(image, image_id, detector_path, crop_factor, catch, verbose):
    This function extracts the subsets that are required to find the fid points in the images.
    These subsets are found using a machine learning approach (with the library dlib). In order to speed up the process,
    before applying the machine learning approach, the image will be initially cropped
    Args:
        image (np-array): The image where subsets should be extracted
        image_id (String): The image_id of the images where subsets should be extracted
        detector_path (String, None): The path where the detector are located. If parameter is 'None' the default
           path is used
        model_name (String, None): The name of the model. If parameter is 'None', the default model is used
        fid_type (Number, None): The image_id of the fid type that should be extracted
        crop_factor (Float, 0.1): How much of the image will be used for the initial crop
        binarize_subset (Boolean): decided if the image will be binary before apply dlib
        overwrite (Boolean): If true, we don't extract fid-marks for already existing fid-marks
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        coords (Dict): dict with 4 entries ("N", "E", "S", "W"), each filled with a list of four coordinates
            x_left, x_right, y_top, y_bottom
    """

    p.print_v(f"Start: extract_subsets ({image_id})", verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get path to folder where the models for the detection of fid_subsets are
    if detector_path is None:
        detector_path = json_data["path_folder_dlib_subset_detectors"]

    # get model name
    if model_name is None:
        model_name = json_data["extract_subsets_detector_name"]

    # get fid type
    if fid_type is None:
        fid_type = json_data["extract_subsets_fid_type"]

    # get crop factor
    if crop_factor is None:
        crop_factor = json_data["extract_subsets_crop_factor"]

    # get already existing subset information from the db
    if overwrite is False:
        sql_string = f"SELECT * FROM images_fid_points WHERE image_id='{image_id}'"
        table_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)
    else:
        table_data = None

    # get size params of the image
    height, width = image.shape
    mid_y = int(height / 2)
    mid_x = int(width / 2)

    # get the subset size in which
    crop_height = int(crop_factor * height)
    crop_width = int(crop_factor * width)

    # create subset_dict (in a subset we will look for the fid points)
    subsets_big = {
        "n": image[0:crop_height, mid_x - crop_width:mid_x + crop_width],
        "e": image[mid_y - crop_height:mid_y + crop_height, width - crop_width:width],
        "s": image[height - crop_height:height, mid_x - crop_width:mid_x + crop_width],
        "w": image[mid_y - crop_height:mid_y + crop_height, 0:crop_width]
    }

    if debug_show_crops:
        di.display_images(list(subsets_big.values()), title=image_id, list_of_titles=["n", "e", "s", "w"])

    # load the detection models
    models = {
        "n": dlib.simple_object_detector(detector_path + "/n_" + model_name + "_" + str(fid_type) + ".svm"),
        "e": dlib.simple_object_detector(detector_path + "/e_" + model_name + "_" + str(fid_type) + ".svm"),
        "s": dlib.simple_object_detector(detector_path + "/s_" + model_name + "_" + str(fid_type) + ".svm"),
        "w": dlib.simple_object_detector(detector_path + "/w_" + model_name + "_" + str(fid_type) + ".svm"),
    }

    # detect and save the coords of the subset that was found by the dlib-detector
    coords = {}
    for key in ["n", "e", "s", "w"]:

        # if we already have the data (and it is not estimated) we don't need to extract data again
        if overwrite is False and \
                table_data[f"subset_{key}_x"].iloc[0] is not None and \
                table_data[f"subset_{key}_estimated"].iloc[0]:
            p.print_v(f"Subset for {key} for {image_id} already estimated", verbose,
                      "green", pbar=pbar)
            continue

        orig_subset = copy.deepcopy(subsets_big[key])
        subset = copy.deepcopy(subsets_big[key])

        if binarize_subset:
            # binarize subset
            window_size = 25
            k = -0.2

            # Calculate local mean and standard deviation using a window of size window_size
            mean = cv2.blur(subset, (window_size, window_size))
            mean_sq = cv2.blur(subset ** 2, (window_size, window_size))
            std = ((mean_sq - mean ** 2) ** 0.5)
            # Compute the threshold using the formula t(x,y) = mean(x,y) + k * std(x,y)
            threshold = mean + k * std
            # Binarize the image using the threshold

            # binarize
            subset[subset < threshold] = 0
            subset[subset != 0] = 255

            kernel1 = np.ones((3, 3), np.uint8)
            subset = cv2.dilate(subset, kernel1, iterations=1)

        # if subset is too big -> resize, otherwise dlib fails
        while subset.shape[0] > 1800 or subset.shape[1] > 1800:
            shape0 = int(subset.shape[0] * 0.9)
            shape1 = int(subset.shape[1] * 0.9)

            subset = cv2.resize(subset, (shape1, shape0), interpolation=cv2.INTER_NEAREST)

        # get difference in x and y
        y_scale = orig_subset.shape[0] / subset.shape[0]
        x_scale = orig_subset.shape[1] / subset.shape[1]

        try:
            # try to find the position of the fid marker
            detection = models[key](subset)
        except (Exception,) as e:
            if catch:

                p.print_v(f"Something went wrong extracting a subset for {image_id} at direction {key}",
                          verbose, pbar=pbar)

                # return empty subset
                coords[key] = None
                continue
            else:
                raise e

        # nothing found
        if len(detection) == 0:
            p.print_v(f"No subset found for {image_id} at direction {key}", verbose,
                      "red", pbar=pbar)
            coords[key] = None

        # too many subsets found
        elif len(detection) > 1:
            p.print_v(f"Too many subsets ({len(detection)}) found for {image_id} at direction {key}",
                      verbose, pbar=pbar)
            coords[key] = None

        # Hooray, a subset was found
        else:
            x_left = int(detection[0].left() * x_scale)
            y_top = int(detection[0].top() * y_scale)
            x_right = int(detection[0].right() * x_scale)
            y_bottom = int(detection[0].bottom() * y_scale)

            coords[key] = [x_left, x_right, y_top, y_bottom]

    if debug_show_subsets_crop:

        # get coords
        bboxes = list(coords.values())

        # adapt coords for visualization
        bboxes_adapted = []
        for elem in bboxes:
            if elem is None:
                bboxes_adapted.append(None)
            else:
                bboxes_adapted.append([elem[0], elem[2], elem[1] - elem[0], elem[3] - elem[2]])

        print(bboxes_adapted)

        di.display_images(list(subsets_big.values()), title=image_id, list_of_titles=["n", "e", "s", "w"],
                          bboxes=bboxes_adapted)

    # translate the coordinates back from crop coordinates to the image coordinates
    for key in ["n", "e", "s", "w"]:

        if coords[key] is None:
            continue

        if key == "n":
            coords[key][0] = coords[key][0] + mid_x - crop_width  # x_left
            coords[key][1] = coords[key][1] + mid_x - crop_width  # x_right
        elif key == "e":
            coords[key][0] = coords[key][0] + width - crop_width  # x_left
            coords[key][1] = coords[key][1] + width - crop_width  # x_right
            coords[key][2] = coords[key][2] + mid_y - crop_height  # y_top
            coords[key][3] = coords[key][3] + mid_y - crop_height  # y_bottom
        elif key == "s":
            coords[key][0] = coords[key][0] + mid_x - crop_width  # x_left
            coords[key][1] = coords[key][1] + mid_x - crop_width  # x_right
            coords[key][2] = coords[key][2] + height - crop_height  # y_top
            coords[key][3] = coords[key][3] + height - crop_height  # y_bottom
        elif key == "w":
            coords[key][2] = coords[key][2] + mid_y - crop_height  # y_top
            coords[key][3] = coords[key][3] + mid_y - crop_height  # y_bottom

    if debug_show_subsets_total:
        # get coords
        bboxes = list(coords.values())

        # adapt coords for visualization
        bboxes_adapted = []
        for elem in bboxes:
            if elem is not None:
                bboxes_adapted.append([elem[0], elem[2], elem[1] - elem[0], elem[3] - elem[2]])
        di.display_images(image, title=image_id, bboxes=[bboxes_adapted])

    p.print_v(f"Finished: extract_subsets ({image_id})", verbose, pbar=pbar)

    return coords
