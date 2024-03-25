import cv2
import dlib
import numpy as np

DETECTOR_PATH = "path/to/detector"
MODEL_NAME = "model_name"
FID_TYPE = "fid_type"
CROP_FACTOR = "crop_factor"


def extract_subset(image, key,
                   detector_path=None, model_name=None,
                   fid_type=None, crop_factor=None, binarize_crop=False):

    detector_path = detector_path or DETECTOR_PATH
    model_name = model_name or MODEL_NAME
    fid_type = fid_type or FID_TYPE
    crop_factor = crop_factor or CROP_FACTOR

    # get size params of the image
    height, width = image.shape
    mid_y = int(height / 2)
    mid_x = int(width / 2)

    # get the crop size in which we look for the subset
    crop_height = int(crop_factor * height)
    crop_width = int(crop_factor * width)

    # create the crop dict
    crop_dict = {
        "n": image[0:crop_height, mid_x - crop_width:mid_x + crop_width],
        "e": image[mid_y - crop_height:mid_y + crop_height, width - crop_width:width],
        "s": image[height - crop_height:height, mid_x - crop_width:mid_x + crop_width],
        "w": image[mid_y - crop_height:mid_y + crop_height, 0:crop_width]
    }

    # load the detection models
    models = {
        "n": dlib.simple_object_detector(detector_path + "/n_" + model_name + "_" + str(fid_type) + ".svm"),
        "e": dlib.simple_object_detector(detector_path + "/e_" + model_name + "_" + str(fid_type) + ".svm"),
        "s": dlib.simple_object_detector(detector_path + "/s_" + model_name + "_" + str(fid_type) + ".svm"),
        "w": dlib.simple_object_detector(detector_path + "/w_" + model_name + "_" + str(fid_type) + ".svm"),
    }

    # already initialize the subset dict
    subset_dict = {
        "n": None,
        "e": None,
        "s": None,
        "w": None
    }

    # iterate over all directions
    for key in ["n", "e", "s", "w"]:

        # get the crop for this key
        crop = crop_dict[key]

        # binarize crop
        if binarize_crop:

            window_size = 25
            k = -0.2

            # Calculate local mean and standard deviation using a window of size window_size
            mean = cv2.blur(crop, (window_size, window_size))
            mean_sq = cv2.blur(crop ** 2, (window_size, window_size))
            std = ((mean_sq - mean ** 2) ** 0.5)

            # Compute the threshold using the formula t(x,y) = mean(x,y) + k * std(x,y)
            threshold = mean + k * std

            # binarize
            crop[crop < threshold] = 0
            crop[crop != 0] = 255

            kernel1 = np.ones((3, 3), np.uint8)
            crop = cv2.dilate(crop, kernel1, iterations=1)

        # get original shape of the crop
        orig_shape = crop.shape

        # if crop is too big -> resize, otherwise dlib fails
        while crop.shape[0] > 1800 or crop.shape[1] > 1800:
            shape0 = int(crop.shape[0] * 0.9)
            shape1 = int(crop.shape[1] * 0.9)

            crop = cv2.resize(crop, (shape1, shape0), interpolation=cv2.INTER_NEAREST)

        # get difference in x and y for crop and resized crop
        y_scale = orig_shape[0] / crop.shape[0]
        x_scale = orig_shape[1] / crop.shape[1]

        try:
            # try to find the position of the fid marker
            detection = models[key](crop)
        except (Exception,):
            continue

        # if no detection or multiple detections -> continue
        if len(detection) != 0:
            continue

        # adapt the position back to the original image
        x_left = int(detection[0].left() * x_scale)
        y_top = int(detection[0].top() * y_scale)
        x_right = int(detection[0].right() * x_scale)
        y_bottom = int(detection[0].bottom() * y_scale)

        # save the coords of the subset
        subset_dict[key] = [x_left, x_right, y_top, y_bottom]

    return subset_dict
