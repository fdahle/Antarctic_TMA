import copy
import dlib
import cv2

import connect_to_db as ctd
import display_multiple_images as dmi
import display_single_image as dsi

path_detection_models = "<Please enter the detector path>"

def extract_subsets(image, image_id=None, detector_path=None, model_name=None, fid_type=None,
                    crop_factor=0.1, catch=True, verbose=False):

    """
    extract_subsets(image, image_id, detector_path, crop_factor, catch, verbose):
    This function extracts the subsets that are required to find the fid points in the images.
    These subsets are found using a machine learning approach (with the library dlib). In order to speed up the process,
    before applying the machine learning approach, the image will be initially cropped
    Args:
        - image (np-array): The image where subsets should be extracted
        - image_id (String, None): If supplied can be used to show as a title in the debugging-figures
        - detector_path (String, None): The path where the detector are located. If parameter is 'None' the default
            path is used
        - mode_name (String, None): The name of the model. If parameter is 'None', the default model is used
        - fid_type (Number, None): The id of the fid type that should be extracted
        - crop_factor (Float, 0.1): How much of the image will be used for the initial crop
        - catch (Boolean, True): If true and something is going wrong, the operation will continue and not crash.
            In this case None is returned
        - verbose (Boolean, False): If true, the status of the operations are printed
    Returns:
        - coords (Dict): dict with 4 entries ("N", "E", "S", "W"), each filled with a list of four coordinates
            x_left, x_right, y_top, y_bottom
    """

    # debug_params
    debug_show_crops = False  # the initial crops
    debug_show_subsets_crop = False  # show the subsets in the crop
    debug_show_subsets_total = False  # show the subsets in the complete image

    # set path to folder where the models for the detection of fid_subsets are
    if detector_path is None:
        detector_path = path_detection_models
        
    if model_name is None:
        model_name = "detector"

    if fid_type is None:
        fid_type = 1

    # get size params of the image
    height, width = image.shape
    mid_y = int(height/2)
    mid_x = int(width/2)

    # get the subset size in which
    crop_height = int(crop_factor * height)
    crop_width = int(crop_factor * width)

    # create subset_dict (in a subset we will look for the fid points)
    subsets_big = {
        "N": image[0:crop_height, mid_x - crop_width:mid_x + crop_width],
        "E": image[mid_y - crop_height:mid_y + crop_height, width - crop_width:width],
        "S": image[height - crop_height:height, mid_x - crop_width:mid_x + crop_width],
        "W": image[mid_y - crop_height:mid_y + crop_height, 0:crop_width]
    }

    if debug_show_crops:
        dmi.display_multiple_images(list(subsets_big.values()), title=image_id, subtitles=["N", "E", "S", "W"])

    # load the detection models
    models = {
        "N": dlib.simple_object_detector(detector_path + "/n_detector_" + str(fid_type) + ".svm"),
        "E": dlib.simple_object_detector(detector_path + "/e_detector_" + str(fid_type) + ".svm"),
        "S": dlib.simple_object_detector(detector_path + "/s_detector_" + str(fid_type) + ".svm"),
        "W": dlib.simple_object_detector(detector_path + "/w_detector_" + str(fid_type) + ".svm"),
    }

    # detect and save the coords of the subset that was found by the dlib-detector
    coords = {}
    for key in ["N", "E", "S", "W"]:

        orig_subset = copy.deepcopy(subsets_big[key])
        subset = copy.deepcopy(subsets_big[key])

        # if subset is too big -> resize, otherwise dlib fails
        while subset.shape[0] > 1800 or subset.shape[1] > 1800:
            shape0 = int(subset.shape[0]*0.9)
            shape1 = int(subset.shape[1]*0.9)

            subset = cv2.resize(subset, (shape1, shape0), interpolation=cv2.INTER_NEAREST)

        # get difference in x and y
        y_scale = orig_subset.shape[0] / subset.shape[0]
        x_scale = orig_subset.shape[1] / subset.shape[1]

        try:
            # try to find the position of the fid marker
            detection = models[key](subset)
        except (Exception,) as e:
            if catch:

                if verbose:
                    print(f"Something went wrong extracting a subset for {image_id} at direction {key}")

                # return empty subset
                coords[key] = None
                continue
            else:
                raise e

        # nothing found
        if len(detection) == 0:
            if verbose:
                print(f"No subset found for {image_id} at direction {key}")
            coords[key] = None

        # too many subsets found
        elif len(detection) > 1:
            if verbose:
                print(f"Too many subsets ({len(detection)}) found for {image_id} at direction {key}")
            coords[key] = None

        # Hooray, a subset was found
        else:
            x_left = int(detection[0].left() * x_scale)
            y_top = int(detection[0].top() * y_scale)
            x_right = int(detection[0].right() * x_scale)
            y_bottom = int(detection[0].bottom() * y_scale)

            coords[key] = [x_left, x_right, y_top, y_bottom]

    if debug_show_subsets_crop:
        dmi.display_multiple_images(list(subsets_big.values()), title=image_id, subtitles=["N", "E", "S", "W"],
                                    bboxes=list(coords.values()))

    # translate the coordinates back from crop coordinates to the image coordinates
    for key in ["N", "E", "S", "W"]:

        if coords[key] is None:
            continue

        if key == "N":
            coords[key][0] = coords[key][0] + mid_x - crop_width  # x_left
            coords[key][1] = coords[key][1] + mid_x - crop_width  # x_right
        elif key == "E":
            coords[key][0] = coords[key][0] + width - crop_width  # x_left
            coords[key][1] = coords[key][1] + width - crop_width  # x_right
            coords[key][2] = coords[key][2] + mid_y - crop_height  # y_top
            coords[key][3] = coords[key][3] + mid_y - crop_height  # y_bottom
        elif key == "S":
            coords[key][0] = coords[key][0] + mid_x - crop_width  # x_left
            coords[key][1] = coords[key][1] + mid_x - crop_width  # x_right
            coords[key][2] = coords[key][2] + height - crop_height  # y_top
            coords[key][3] = coords[key][3] + height - crop_height  # y_bottom
        elif key == "W":
            coords[key][2] = coords[key][2] + mid_y - crop_height  # y_top
            coords[key][3] = coords[key][3] + mid_y - crop_height  # y_bottom

    if debug_show_subsets_total:
        dsi.display_single_image(img, title=image_id, bbox=list(coords.values()))

    return coords


def train_subset_extractor(num_images=250, model_name="model", model_path=None):

    """
    train_subset_extractor(num_images):
    This function trains the detector for the subsets and saves it to the folder. From the database a certain amount of
    random example images is loaded together with the subset coordinates for the training
    Args:
        - num_images (int, 250): With how many examples should be trained
        - model_name(str, "model_"): What is the name of the model (with "N", "E", "S", "W" at the end attached)
        - model_path(str, None): Where should the model be stored. If none the default path is used
    Returns:
        None
    """

    directions = ["n", "s", "e", "w"]

    if model_path is None:
        model_path = "/media/fdahle/beb5a64a-5335-424a-8f3c-779527060523/ATM/data/machine_learning/subsets"

    for direction in directions:

        # get all subset coordinates
        sql_string = f"SELECT image_id, subset_{direction}_x, subset_{direction}_y, subset_width, subset_height " \
                     f"from images_properties WHERE " \
                     f"subset_{direction}_x IS NOT NULL and subset_{direction}_y IS NOT NULL"

        data = ctd.get_data_from_db(sql_string, catch=False)

        crops = []
        boxes = []

        # shuffle the data
        data = data.sample(frac=1).reset_index(drop=True)

        for index, row in data.iterrows():

            print(f"Load {row['image_id']}")

            if index == num_images:
                break

            image = liff.load_image_from_file(row["image_id"])

            min_y = int(row[f"subset_{direction}_y"])
            max_y = int(row[f"subset_{direction}_y"] + row["subset_height"])
            min_x = int(row[f"subset_{direction}_x"])
            max_x = int(row[f"subset_{direction}_x"] + row["subset_width"])

            if min_y < 0:
                print("WARNING")
                min_y = 0
            if min_x < 0:
                print("WARNING")
                min_x = 0

            subset_factor = 0.1

            # get size params of the image
            height, width = image.shape
            mid_y = int(height / 2)
            mid_x = int(width / 2)
            subset_height = int(subset_factor * height)
            subset_width = int(subset_factor * width)

            # TODO workaround
            if direction == "n" and min_y > 1000:
                continue
            elif direction == "s" and min_y < 8750:
                continue

            # init crop so that ide is not complaining
            crop = None

            if direction == "n":
                crop = image[0:subset_height, mid_x - subset_width:mid_x + subset_width]
                min_x = min_x - (mid_x - subset_width)
                max_x = max_x - (mid_x - subset_width)
            elif direction == "e":
                crop = image[mid_y - subset_height:mid_y + subset_height, width - subset_width:width]
                min_x = min_x - (width-subset_width)
                max_x = max_x - (width-subset_width)
                min_y = min_y - (mid_y - subset_height)
                max_y = max_y - (mid_y - subset_height)
            elif direction == "s":
                crop = image[height - subset_height:height, mid_x - subset_width:mid_x + subset_width]
                min_x = min_x - (mid_x - subset_width)
                max_x = max_x - (mid_x - subset_width)
                min_y = min_y - (height - subset_height)
                max_y = max_y - (height - subset_height)
            elif direction == "w":
                crop = image[mid_y - subset_height:mid_y + subset_height, 0:subset_width]
                min_y = min_y - (mid_y - subset_height)
                max_y = max_y - (mid_y - subset_height)

            crops.append(crop)

            box = [dlib.rectangle(left=min_x, top=min_y, right=max_x, bottom=max_y)]
            boxes.append(box)

        print(f"Train detector for {direction}")

        # set options for dlib
        options = dlib.simple_object_detector_training_options()
        options.add_left_right_image_flips = False
        options.C = 5
        options.num_threads = 10
        options.be_verbose = True

        detector = dlib.train_simple_object_detector(crops, boxes, options)
        file_name = model_path + "/" + model_name + "_" + direction + ".svm"

        detector.save(file_name)

