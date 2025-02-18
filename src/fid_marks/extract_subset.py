"""extract a subset for fiducial marker detection"""

# Library imports
import cv2
import dlib
import numpy as np
from typing import Optional

import src.display.display_images as di

# Constants
DETECTOR_PATH = "/data/ATM/data_1/machine_learning/dlib/subsets"
MODEL_NAME = "detector"
FID_TYPE = 1
CROP_FACTOR = 0.1


DEBUG_SHOW_CROP = True
DEBUG_SHOW_MULTI_DETECTIONS = True

def extract_subset(image: np.ndarray, key: str,
                   detector_path: Optional[str] = None, model_name: Optional[str] = None,
                   fid_type: Optional[str] = None, crop_factor: Optional[float] = None,
                   refine_multiple: bool = False,
                   binarize_crop: bool = False,
                   catch=True) -> Optional[list[int]]:
    """
    Extracts a subset of an image in which a fid mark for a key direction ('n', 'e', 's', 'w')
    is located. The detection is using a pre-trained model. The bounding box of the subset
        is returned.
    Args:
        image (np.ndarray): The input image from which a subset will be extracted.
        key (str): The key indicating the direction of the subset to extract ('n', 'e', 's', 'w').
        detector_path (Optional[str]): The file system path to the directory containing
            the detection models. Defaults to a predefined path.
        model_name (Optional[str]): The name of the model to be used for detection.
            Defaults to a predefined model name.
        fid_type (Optional[str]): The type of fiducial marker to be detected.
            Defaults to a predefined type.
        crop_factor (Optional[float]): The factor by which the image is cropped.
            Defaults to a predefined factor.
        binarize_crop (bool): If True, the cropped image will be binarized before detection.
            Defaults to False.
    Returns:
        Optional[List[int]]: The bounding box [x_left, x_right, y_top, y_bottom] of the
            detected fiducial marker, or None (if no marker is detected or if there are
            multiple detections).
    """

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

    if key == "n":
        crop = image[0:crop_height, mid_x - crop_width:mid_x + crop_width]
    elif key == "e":
        crop = image[mid_y - crop_height:mid_y + crop_height, width - crop_width:width]
    elif key == "s":
        crop = image[height - crop_height:height, mid_x - crop_width:mid_x + crop_width]
    elif key == "w":
        crop = image[mid_y - crop_height:mid_y + crop_height, 0:crop_width]
    else:
        raise ValueError("Key must be one of 'n', 'e', 's', 'w'.")

    # load the detection model
    model = dlib.simple_object_detector(detector_path + f"/{key}_" + model_name + "_" + str(fid_type) + ".svm")

    # binarize crop
    if binarize_crop:

        print("WARNING: NOT WORKING WITH DLIB")

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
    catch = False

    if DEBUG_SHOW_CROP:
        di.display_images([crop])

    try:
        # try to find the position of the fid marker
        detections = model(crop)

    except (Exception,) as e:
        if catch:
            return None
        else:
            raise e

    # if no detection -> continue
    if len(detections) == 0:
        return None

    # if multiple detections -> continue
    if len(detections) > 1:

        if DEBUG_SHOW_MULTI_DETECTIONS:
            # convert detection to list of coords
            bboxes = [[d.left(), d.top(), d.right(), d.bottom()] for d in detections]
            di.display_images([crop], bounding_boxes=[bboxes])

        if refine_multiple is False:
                return None
        else:
            # get the largest detection
            ld = max(detections, key=lambda d: (d.right() - d.left()) * (d.bottom() - d.top()))
            detections = [ld]

    # adapt the position back to the original image
    x_left = int(detections[0].left() * x_scale)
    y_top = int(detections[0].top() * y_scale)
    x_right = int(detections[0].right() * x_scale)
    y_bottom = int(detections[0].bottom() * y_scale)

    # translate the coordinates back from crop coordinates to the image coordinates
    if key == "n":
        x_left = x_left + mid_x - crop_width
        x_right = x_right + mid_x - crop_width
    elif key == "e":
        x_left = x_left + width - crop_width
        x_right = x_right + width - crop_width
        y_top = y_top + mid_y - crop_height
        y_bottom = y_bottom + mid_y - crop_height
    elif key == "s":
        x_left = x_left + mid_x - crop_width
        x_right = x_right + mid_x - crop_width
        y_top = y_top + height - crop_height
        y_bottom = y_bottom + height - crop_height
    elif key == "w":
        y_top = y_top + mid_y - crop_height
        y_bottom = y_bottom + mid_y - crop_height

    # save the coords of the subset
    bounds = [x_left, x_right, y_top, y_bottom]

    return bounds


if __name__ == "__main__":

    id = "CA173632V0009"
    keys = ['n', 'e', 's', 'w']

    import src.load.load_image as li
    img = li.load_image(id)

    di.display_images([img])
    import src.base.rotate_image as ri
    img = ri.rotate_image(img, 180)

    for key in keys:
        bounds = extract_subset(img, key, refine_multiple=True)
        print(bounds)
