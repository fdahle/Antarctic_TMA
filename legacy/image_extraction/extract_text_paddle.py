import argparse
import copy
import cv2
import json

import numpy as np
import os
import pandas as pd
import shapely.geometry

from paddleocr import PaddleOCR, draw_ocr  # noqa

import base.load_image_from_file as liff
import base.print_v as p

import display.display_images as di

debug_keys = ["n", "e", "s", "w"]
debug_show_small_subsets = False  # show the subsets per direction (normal and up-side down)
debug_show_small_subsets_polygons = False
debug_show_subsets_polygons_unmerged = False
debug_show_subset_polygons = False  # show the found text-polygons per subset
debug_show_all_polygons = False
debug_show_all_merged_polygons = False
debug_print_raw_data_all = False
debug_print_raw_data_final = False


# textbox is [min_x_abs, min_y_abs, max_x_abs, max_y_abs]

def extract_text_paddle(img_id, binarize_image_str=None,
                        subset_height=None, nr_small_subsets=None, subset_overlap=None,
                        min_confidence=None, min_area=None,
                        padding_x=None, padding_y=None,
                        catch_str=True, verbose_str=False, pbar=None):
    """
    extract_text_paddle(img, image_id, return_bounds, binarize_image, min_confidence,
                 min_area, padding, catch, verbose, pbar):
    This function is looking for text on the image and trying to recognize it (ocr). Note that
    this function is only looking at the edges of the images and not for text in the center of
    the image. Additionally, it can return bounding boxes for where the text is located.
    Args:
        img_id (String): The image-image_id of the image in which we are looking for text
        binarize_image_str (Bool): If true, the image will be converted into a binary image
            before text is extract. Usually this improves the text extraction.
        subset_height (Integer):
        nr_small_subsets (Integer):
        subset_overlap (Integer):

        min_confidence (Integer): The minimum confidence for a recognized text. 0 means no
            confidence, 100 means absolute confidence.
        min_area (Integer): The minimum area in pixels, that the polygon around a recognized
            text must have. This helps to remove small text-boxes with only one letter.
        padding_x (Integer): How many pixels will be added at every horizontal side of the
            polygon of a recognized text
        padding_y (Integer): Same as padding_x, but for the vertical side
        catch_str (Boolean): If true, we catch every error that is happening and return instead None
        verbose_str (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        text_content:
        text_bounds:
        text_conf:
    """

    # get some parameters again
    img = liff.load_image_from_file(img_id)
    binarize_image = False if binarize_image_str == "False" else True
    catch = False if catch_str == "False" else True
    verbose = False if verbose_str == "False" else True

    p.print_v(f"Start: extract_text_paddle ({img_id})", verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the subset height
    if subset_height is None:  # how high is the subset?
        subset_height = json_data["extract_text_subset_height"]

    # get nr of small subsets
    if nr_small_subsets is None:
        nr_small_subsets = json_data["extract_text_nr_small_subsets"]

    # get overlap between subsets
    if subset_overlap is None:
        subset_overlap = json_data["extract_text_subset_overlap"]

    if binarize_image is None:
        binarize_image = json_data["extract_text_binarize_img"]

    if min_confidence is None:
        min_confidence = json_data["extract_text_min_conf"]

    if min_area is None:
        min_area = json_data["extract_text_min_area"]

    if padding_x is None:
        padding_x = json_data["extract_text_padding_x"]

    if padding_y is None:
        padding_y = json_data["extract_text_padding_y"]

    # copy image so that we don't change the original
    img = copy.deepcopy(img)

    try:
        # init ocr
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, version="PP-OCR", use_gpu=False)

        text_position_all = []
        text_content_all = []
        text_conf_all = []

        # text is usually only at the edges of the images, so we need do this process 4 times
        for key in ["n", "e", "s", "w"]:

            # get subset of images based on the direction (makes it easier to find the text)
            if key == "n":
                subset_orig = img[:subset_height, 500:img.shape[1] - 500]
            elif key == "e":
                subset_orig = img[:, img.shape[1] - subset_height:]
            elif key == "s":
                subset_orig = img[img.shape[0] - subset_height:, 500:img.shape[1] - 500]
            elif key == "w":
                subset_orig = img[:, :subset_height]

            # copy subset to not change something in the img
            subset = copy.deepcopy(subset_orig)  # noqa

            # we want all subsets with the text written from left to right
            if key == "e":
                subset = np.rot90(subset, 1)
            elif key == "s":
                subset = np.rot90(subset, 2)
            elif key == "w":
                subset = np.rot90(subset, 3)

            # sometimes it is difficult to find the text in the image (due to the background)
            # -> binarize the image
            if binarize_image:

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
                subset[subset != 0] = 1
                subset = 1 - subset

                kernel1 = np.ones((3, 3), np.uint8)
                subset = cv2.dilate(subset, kernel1, iterations=1)

                subset = 1 - subset

                contours, hierarchy = cv2.findContours(subset, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cv2.fillPoly(subset, [cnt], 255)

                subset[subset > 0] = 1

                subset = 1 - subset

                # subset = cv2.GaussianBlur(subset, (5, 5), 0)

                # erode and dilate to make it smoother
                # kernel1 = np.ones((3, 3), np.uint8)
                # subset = cv2.erode(subset, kernel1, iterations=1)
                # subset = cv2.dilate(subset, kernel1, iterations=1)

            # how big is a small subset
            subset_small_width = int(subset.shape[1] / nr_small_subsets)

            # here we save all data for subset 1 and subset 2
            data_1 = None
            data_2 = None

            # iterate all small subsets
            for i in range(nr_small_subsets):

                # get the position and extent of the small subset
                small_subset_left = i * subset_small_width
                small_subset_right = (i + 1) * subset_small_width

                small_subset_left = small_subset_left - subset_overlap
                small_subset_left = max(0, small_subset_left)
                small_subset_right = small_subset_right - subset_overlap
                small_subset_right = min(subset.shape[1], small_subset_right)

                small_subset = subset[:, small_subset_left:small_subset_right]

                # as text can be written upside-down (they really didn't handle these images good...)
                # we want to look at every subset twice: normal and upside-down

                # first the subset we are not rotating, then the rotated one
                small_subset_1 = copy.deepcopy(small_subset)
                small_subset_2 = np.rot90(copy.deepcopy(small_subset), 2)

                if debug_show_small_subsets and key in debug_keys:
                    di.display_images([small_subset_1, small_subset_2],
                                      title=f"Subsets ({i + 1}/{nr_small_subsets}) for '{key}'")

                # here we save the boxes and the text we found
                text_pos_rel = []
                text_pos_abs = []
                text_content = []
                text_conf = []

                # find text in image
                small_data_1 = ocr.ocr(small_subset_1)
                small_data_2 = ocr.ocr(small_subset_2)

                list_1 = []
                for elem in small_data_1[0]:
                    bbox = elem[0]
                    left = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
                    right = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
                    top = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
                    bottom = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
                    height = bottom - top
                    width = right - left
                    text = elem[1][0]
                    conf_rounded = elem[1][1] * 100
                    new_row = {'left': left, 'right': right, 'top': top, 'bottom': bottom,
                               'height': height, 'width': width, 'text': text, 'conf': conf_rounded}
                    list_1.append(new_row)
                if len(list_1) == 0:
                    small_data_1 = pd.DataFrame(columns=['left', 'right', 'top', 'bottom',
                                                         'height', 'width', 'text', 'conf'])
                else:
                    small_data_1 = pd.DataFrame.from_dict(list_1)  # noqa

                list_2 = []
                for elem in small_data_2[0]:
                    bbox = elem[0]
                    left = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
                    right = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
                    top = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
                    bottom = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
                    height = bottom - top
                    width = right - left
                    text = elem[1][0]
                    conf_rounded = elem[1][1] * 100
                    new_row = {'left': left, 'right': right, 'top': top, 'bottom': bottom,
                               'height': height, 'width': width, 'text': text, 'conf': conf_rounded}
                    list_2.append(new_row)
                if len(list_2) == 0:
                    small_data_2 = pd.DataFrame(columns=['left', 'right', 'top', 'bottom',
                                                         'height', 'width', 'text', 'conf'])
                else:
                    small_data_2 = pd.DataFrame.from_dict(list_2)  # noqa

                if debug_print_raw_data_all and key in debug_keys:
                    print(small_data_1)
                    print(small_data_2)

                # remove -1 conf values
                small_data_1 = small_data_1[small_data_1['conf'] != -1]
                small_data_2 = small_data_2[small_data_2['conf'] != -1]

                # we only want values with a higher confidence
                small_data_1 = small_data_1.loc[small_data_1['conf'] >= min_confidence]
                small_data_2 = small_data_2.loc[small_data_2['conf'] >= min_confidence]

                # already remove empty text values
                small_data_1['text'] = small_data_1['text'].apply(lambda x: str(x).strip())
                small_data_1 = small_data_1[small_data_1['text'] != ""]
                small_data_2['text'] = small_data_2['text'].apply(lambda x: str(x).strip())
                small_data_2 = small_data_2[small_data_2['text'] != ""]

                if debug_show_small_subsets_polygons and key in debug_keys:
                    boxes1 = zip(small_data_1['left'], small_data_1['top'],
                                 small_data_1['width'], small_data_1['height'])
                    boxes2 = zip(small_data_2['left'], small_data_2['top'],
                                 small_data_2['width'], small_data_2['height'])
                    boxes1 = [list(x) for x in boxes1]
                    boxes2 = [list(x) for x in boxes2]

                    di.display_images([small_subset_1, small_subset_2],
                                      bboxes=[boxes1, boxes2])

                # rotate the points back for small data 2
                for index, row in small_data_2.iterrows():
                    left, top = rot90points(row['left'], row['top'], -2, small_subset_2.shape)
                    small_data_2.loc[index, 'left'] = left - row['width']
                    small_data_2.loc[index, 'top'] = top - row['height']

                # change left to the absolute subset left
                small_data_1['left'] = small_data_1['left'] + i * subset_small_width
                small_data_2['left'] = small_data_2['left'] + i * subset_small_width

                # add right and bottom to the data
                small_data_1['right'] = small_data_1['left'] + small_data_1['width']
                small_data_1['bottom'] = small_data_1['top'] + small_data_1['height']
                small_data_2['right'] = small_data_2['left'] + small_data_2['width']
                small_data_2['bottom'] = small_data_2['top'] + small_data_2['height']

                # merge the dataframes
                if data_1 is None:
                    data_1 = small_data_1
                else:
                    data_1 = pd.concat([data_1, small_data_1])
                if data_2 is None:
                    data_2 = small_data_2
                else:
                    data_2 = pd.concat([data_2, small_data_2])

            # get the average conf values
            conf_1 = data_1['conf'].mean()
            conf_2 = data_2['conf'].mean()

            if conf_1 >= conf_2:
                data = data_1
            else:
                data = data_2

            if debug_show_subsets_polygons_unmerged and key in debug_keys:
                boxes = zip(data['left'], data['top'],
                            data['width'], data['height'])
                boxes = [list(x) for x in boxes]

                di.display_images(subset, bboxes=[boxes])

            lines = data['text'].tolist()
            confs = data['conf'].tolist()
            left = data['left'].tolist()
            right = data['right'].tolist()
            top = data['top'].tolist()
            bottom = data['bottom'].tolist()

            if debug_print_raw_data_final and key in debug_keys:
                print("Lines:")
                print(lines)
                print("Conf:")
                print(confs)
                print("Position:")
                print(left, right, top, bottom)

            # iterate all found text boxes
            for i in range(len(lines)):

                # we didn't find any text -> continue
                if len(lines[i]) == 0:
                    continue

                # if text is only space -> continue
                if lines[i].isspace():
                    continue

                # get min and max values for x and y
                (min_x_rel, min_y_rel, max_x_rel, max_y_rel) = (left[i], top[i], right[i], bottom[i])

                # convert to polygon
                poly_rel = shapely.geometry.box(min_x_rel, min_y_rel, max_x_rel, max_y_rel)

                # check for a minimum size of the polygon
                if min_area is not None:
                    if poly_rel.area < min_area:
                        continue

                # copy coords for the abs values
                min_x_abs = copy.deepcopy(min_x_rel)
                max_x_abs = copy.deepcopy(max_x_rel)
                min_y_abs = copy.deepcopy(min_y_rel)
                max_y_abs = copy.deepcopy(max_y_rel)

                # rotate points back
                if key == "e":
                    min_x_abs, min_y_abs = rot90points(min_x_rel, min_y_rel, -1, subset.shape)
                    max_x_abs, max_y_abs = rot90points(max_x_rel, max_y_rel, -1, subset.shape)
                elif key == "s":
                    min_x_abs, min_y_abs = rot90points(min_x_rel, min_y_rel, -2, subset.shape)
                    max_x_abs, max_y_abs = rot90points(max_x_rel, max_y_rel, -2, subset.shape)
                elif key == "w":
                    min_x_abs, min_y_abs = rot90points(min_x_rel, min_y_rel, -3, subset.shape)
                    max_x_abs, max_y_abs = rot90points(max_x_rel, max_y_rel, -3, subset.shape)

                # through the rotating sometimes min and max can be switched -> correct
                if max_x_abs < min_x_abs:
                    temp = copy.deepcopy(min_x_abs)
                    min_x_abs = max_x_abs
                    max_x_abs = temp
                if max_y_abs < min_y_abs:
                    temp = copy.deepcopy(min_y_abs)
                    min_y_abs = max_y_abs
                    max_y_abs = temp

                # convert relative to absolute coords
                if key == "n":
                    min_x_abs = min_x_abs + 500
                    max_x_abs = max_x_abs + 500
                elif key == "e":
                    min_x_abs = min_x_abs + img.shape[1] - subset_height
                    max_x_abs = max_x_abs + img.shape[1] - subset_height
                elif key == "s":
                    min_x_abs = min_x_abs + 500
                    max_x_abs = max_x_abs + 500
                    min_y_abs = min_y_abs + img.shape[0] - subset_height
                    max_y_abs = max_y_abs + img.shape[0] - subset_height
                elif key == "w":
                    pass  # no changes required

                # apply padding to the polygon boxes
                min_x_abs = min_x_abs - padding_x
                max_x_abs = max_x_abs + padding_x
                min_y_abs = min_y_abs - padding_y
                max_y_abs = max_y_abs + padding_y

                # check if we are still in the boundary of the polygon
                if min_x_abs < 0:
                    min_x_abs = 0
                if max_x_abs > img.shape[1]:
                    max_x_abs = img.shape[1]
                if min_y_abs < 0:
                    min_y_abs = 0
                if max_y_abs > img.shape[0]:
                    max_y_abs = img.shape[0]

                poly_abs = shapely.geometry.box(min_x_abs, min_y_abs, max_x_abs, max_y_abs)

                # save in list
                text_pos_rel.append(poly_rel)  # noqa
                text_pos_abs.append(poly_abs)  # noqa
                text_content.append(lines[i])  # noqa
                text_conf.append(confs[i])  # noqa

            # check if we have found something
            if len(text_pos_rel) == 0:
                p.print_v(f"No text could be found for subset '{key}'", verbose, pbar=pbar)
            else:
                # show the polygons we've found in the subset
                if debug_show_subset_polygons and key in debug_keys:
                    di.display_images(subset, polygons=[text_pos_rel])

                # yeah, we've found something
                p.print_v(f"{len(text_pos_rel)} polygons found for subset '{key}'", verbose, pbar=pbar)

                # save the stuff we found per subset to a global list
                text_position_all.append(text_pos_abs)
                text_content_all.append(text_content)
                text_conf_all.append(text_conf)

        if debug_show_all_polygons:
            di.display_images(img, polygons=[text_pos_abs])

        # flatten list
        text_content_all = [item for sublist in text_content_all for item in sublist]  # noqa
        text_position_all = [item for sublist in text_position_all for item in sublist]
        text_conf_all = [item for sublist in text_conf_all for item in sublist]  # noqa

        # merge all polygons
        merged_text_positions = []
        for polygon in text_position_all:
            if not merged_text_positions:
                merged_text_positions.append(polygon)
            else:
                # check if the current polygon overlaps with any of the merged polygons
                overlaps = [poly for poly in merged_text_positions if poly.intersects(polygon)]
                if overlaps:
                    # if overlapping polygons -> merge
                    merged = polygon.union(shapely.geometry.MultiPolygon(overlaps))
                    merged_text_positions = [poly for poly in merged_text_positions if poly not in overlaps]
                    merged_text_positions.append(merged)
                else:
                    merged_text_positions.append(polygon)

        if debug_show_all_merged_polygons:
            di.display_images(img, polygons=[merged_text_positions])

        # look for text in the bounding boxes of these merged polygons
        final_text_content = []
        final_pos = []
        final_confs = []
        for merged_poly in merged_text_positions:
            mb = merged_poly.bounds
            merged_box = img[int(mb[1]):int(mb[3]), int(mb[0]):int(mb[2])]
            ocr_results = ocr.ocr(merged_box)
            merged_text = ""
            temp_confs = []
            for elem in ocr_results[0]:
                merged_text = merged_text + elem[1][0] + " "
                temp_confs.append(elem[1][1] * 100)
            merged_text = merged_text[:-1]
            final_text_content.append(merged_text)
            final_confs.append(np.mean(np.asarray(temp_confs)))
            final_pos.append(mb)

        p.print_v(f"Finished: extract_text_paddle ({image_id})", verbose, pbar=pbar)
    except (Exception,) as e:
        if catch:
            p.print_v(f"Failed: extract_text_paddle ({image_id})", verbose, pbar=pbar)
            return None, None, None
        else:
            raise e

    return final_text_content, final_pos, final_confs


# function to rotate points for 90 degrees
def rot90points(in_x, in_y, k, hw):
    x = copy.deepcopy(in_x)
    y = copy.deepcopy(in_y)

    k = k % 4
    if k == 0:
        return x, y
    elif k == 1:
        return y, hw[1] - 1 - x
    elif k == 2:
        return hw[1] - 1 - x, hw[0] - 1 - y
    elif k == 3:
        return hw[0] - 1 - y, x
    else:
        raise ValueError(f"k error {k}")


if __name__ == "__main__":

    image_id = "CA182331L0003"

    # we need to parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--function_name", type=str, default="main",
                        help="Name of the function to call")
    parser.add_argument("--image_id", type=str, default=None,
                        help="Image ID")
    parser.add_argument("--binarize_image_str", type=str, default="False",
                        help="Whether to binarize the image")
    parser.add_argument("--catch_str", type=str, default="True",
                        help="Whether to catch exceptions")
    parser.add_argument("--verbose_str", type=str, default="False",
                        help="Whether to print verbose output")
    args = parser.parse_args()

    # save arguments in variables (some are somehow tuples)
    if args.image_id is not None:
        image_id = args.image_id,
        image_id = image_id[0]   # it is somehow a tuple
    _binarize_image_str = args.binarize_image_str,
    _binarize_image_str = _binarize_image_str[0]
    _catch_str = args.catch_str,
    _catch_str = _catch_str[0]
    _verbose_str = args.verbose_str

    # the actual text finding
    content, pos, conf = extract_text_paddle(image_id,
                                             binarize_image_str=_binarize_image_str,
                                             catch_str=_catch_str,
                                             verbose_str=_verbose_str)

    # IMPORTANT: We need to print this is order to extract it in our main method
    print(content)
    print(pos)
    print(conf)
