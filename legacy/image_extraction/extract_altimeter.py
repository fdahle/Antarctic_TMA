import cv2
import dlib
import json
import numpy as np
import os

import base.print_v as p

import display.display_images as di

debug_show_altimeter_subset = False
debug_show_circle_binary = False
debug_show_circle_in_subset = False
debug_show_circle_total = False


def extract_altimeter(image, image_id,
                      matching_confidence_value=None, min_binary_threshold=None,
                      path_detector_model=None, path_templates=None,
                      catch=True, verbose=False, pbar=None):
    """
    extract_altimeter(image, image_id, matching_confidence_value, min_binary_threshold,
                      path_detector_model, path_templates, catch, verbose, pbar):
    In order to extract the height for an image, we first need to detect where exactly in the
    image the altimeter is located. Furthermore, we need the exact extent of this altimeter. In
    this function, dlib is used to get a subset around the altimeter and the exact position of the
    circle that describes the altimeter.
    Args:
        image (np-array): The image for which we want to extract the altimeter
        image_id (String): The image-image_id of the image
        matching_confidence_value (float): The minimum confidence to find a circle. As higher,
            as better the quality of the found circles, but fewer circles are found
        min_binary_threshold (int): We are binarizing the images for better circle extraction.
            This number tells when the image value should be 0 or 1
        path_detector_model (string): Where is the file containing the dlib-detector (cor the round circle)
        path_templates (string): In order to account for a rotation of the circle we are matching
            the location of different numbers (e.g 3, 5, 8) to find the exact rotation. In this
            folder are the templated located
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of
            the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        bounding_box (list): The bounding box in which the circle is located [min_x, max_x, min_y, max_y]
        circle (tuple): The position of the circle with (x, y, radius)
            (based on absolute values of the image and not on the subset)

    """

    p.print_v(f"Start: extract_altimeter ({image_id})", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if path_detector_model is None:
        path_detector_model = json_data["path_file_dlib_clock_detectors"]

    if path_templates is None:
        path_templates = json_data["path_folder_altimeter_templates"]

    if matching_confidence_value is None:
        matching_confidence_value = json_data["extract_altimeter_matching_conf_val"]

    if min_binary_threshold is None:
        min_binary_threshold = json_data["extract_altimeter_min_binary_threshold"]

    # function to make an image binary
    def image_to_binary(_img, _min_th=0, _max_th=255):
        # converse the image to binary
        # thresholding
        ret, o1 = cv2.threshold(_img, _min_th, _max_th, cv2.THRESH_BINARY)
        # Erosion & Dilation
        kernel1 = np.ones((3, 3), np.uint8)
        img_erosion = cv2.erode(o1, kernel1, iterations=2)
        img_dilation = cv2.dilate(img_erosion, kernel1, iterations=2)
        # white black color exchange()
        img_binary = 255 - img_dilation

        return img_binary

    # function to locate a circle even more exact with template matching
    def locate_circle(_img_binary, _template_fld):
        template3 = cv2.imread(_template_fld + "/3.jpg", 0)
        template5 = cv2.imread(_template_fld + "/5.jpg", 0)
        template8 = cv2.imread(_template_fld + "/8.jpg", 0)

        # width and height of the 3 templates
        w3, h3 = template3.shape[::-1]
        w5, h5 = template5.shape[::-1]
        w8, h8 = template8.shape[::-1]

        # matching for 3
        res3 = cv2.matchTemplate(_img_binary, template3, cv2.TM_CCOEFF)
        min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
        # matching for 5
        res5 = cv2.matchTemplate(_img_binary, template5, cv2.TM_CCOEFF)
        min_val5, max_val5, min_loc5, max_loc5 = cv2.minMaxLoc(res5)
        # matching for 8
        res8 = cv2.matchTemplate(_img_binary, template8, cv2.TM_CCOEFF)
        min_val8, max_val8, min_loc8, max_loc8 = cv2.minMaxLoc(res8)

        r = 168 + 25 + 74
        if max_val3 > matching_confidence_value and max_val3 > max_val5 and max_val3 > max_val8:
            # print('max_val3:',max_val3)
            top_left3 = max_loc3
            bottom_right3 = (top_left3[0] + w3, top_left3[1] + h3)
            mid_p = (int((top_left3[0] + bottom_right3[0]) / 2), int((top_left3[1] + bottom_right3[1]) / 2))
            _circle = (
                int(mid_p[0] - np.cos(18 * np.pi / 180) * 205), int(mid_p[1] - np.sin(18 * np.pi / 180) * 205), r)
        elif max_val5 > matching_confidence_value and max_val5 > max_val3 and max_val5 > max_val8:
            # print('max_val5:', max_val5)
            top_left5 = max_loc5
            bottom_right5 = (top_left5[0] + w5, top_left5[1] + h5)
            _circle = (int((top_left5[0] + bottom_right5[0]) / 2) - 7, top_left5[1] - 169, r)
        elif max_val8 > matching_confidence_value and max_val8 > max_val3 and max_val8 > max_val5:
            # print('max_val8:',max_val8)
            top_left8 = max_loc8
            bottom_right8 = (top_left8[0] + w8, top_left8[1] + h8)
            mid_p = (int((top_left8[0] + bottom_right8[0]) / 2), int((top_left8[1] + bottom_right8[1]) / 2))
            _circle = (
                int(mid_p[0] + np.cos(18 * np.pi / 180) * 205), int(mid_p[1] + np.sin(18 * np.pi / 180) * 205), r)
        else:
            return None

        return _circle

    try:

        # simple object detector for altimeter detection
        detector = dlib.simple_object_detector(path_detector_model)

        # detect where the altimeter is
        detections = detector(image)

        # check if we have only one circle
        if len(detections) == 0:
            p.print_v(f"No altimeters could be detected in {image_id}", verbose, pbar=pbar)
            return None, None
        elif len(detections) > 1:
            p.print_v(f"Too many altimeters detected in {image_id}", verbose, pbar=pbar)
            return None, None

        # get the detection
        d = detections[0]

        # get the bounding box from the detection
        bounding_box = [d.left(), d.right(), d.top(), d.bottom()]

        # assure no values are outside the image
        if bounding_box[0] < 0:
            bounding_box[0] = 0
        if bounding_box[1] > image.shape[1]:
            bounding_box[1] = image.shape[1]
        if bounding_box[2] < 0:
            bounding_box[2] = 0
        if bounding_box[3] > image.shape[0]:
            bounding_box[3] = image.shape[0]

        # get the subset
        altimeter_subset = image[bounding_box[2]:bounding_box[3], bounding_box[0]: bounding_box[1]]

        p.print_v(f"Altimeter detected at: {bounding_box}",
                  verbose, pbar=pbar)

        # show altimeter subset
        if debug_show_altimeter_subset:
            di.display_images(altimeter_subset, title="Altimeter subset")

        # make variable-name shorter
        min_th = min_binary_threshold
        max_th = 255

        # init circle already
        circle = None

        # find a circle based on a binary image (the minimum values for binary are changing)
        for r_min_th in range(min_th, 250, 5):

            # make image binary
            altimeter_subset_binary = image_to_binary(altimeter_subset, r_min_th, max_th)

            # show the binary circle if wished
            if debug_show_circle_binary:
                di.display_images([altimeter_subset, altimeter_subset_binary], title="Circle binary")

            # locate the circle in the subset (x,y format)
            circle = locate_circle(altimeter_subset_binary, path_templates)

            if circle is None:
                p.print_v(f"No circle found with binary ({min_th}, {max_th})", verbose, pbar=pbar)
            if circle is not None:
                p.print_v(f"Circle found at {circle}", verbose, pbar=pbar)
                break

        # we never found a circle
        if circle is None:
            p.print_v(f"Failed: extract_altimeter ({image_id})", verbose=verbose, pbar=pbar)
            return None, None
        else:

            if debug_show_circle_in_subset:
                di.display_images(altimeter_subset,
                                  points=[circle[0], circle[1]], point_size=circle[2],
                                  title="Circle location in subset")

            # convert point coordinates to absolute coordinates
            circle = list(circle)
            circle[0] = circle[0] + bounding_box[0]  # noqa
            circle[1] = circle[1] + bounding_box[2]
            circle = tuple(circle)

            if debug_show_circle_total:
                di.display_images(image,
                                  points=[circle[0], circle[1]], point_size=circle[2],
                                  title="Circle location absolute")

            p.print_v(f"Finished: extract_altimeter ({image_id})", verbose=verbose, pbar=pbar)
    except (Exception,) as e:
        if catch:
            p.print_v(f"Failed: extract_altimeter ({image_id})", verbose=verbose, pbar=pbar)
            return None, None
        else:
            raise e

    return bounding_box, circle
