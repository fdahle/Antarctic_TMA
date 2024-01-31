import copy
import cv2
import json
import numpy as np
import os
import pandas as pd
import pytesseract
import shapely.geometry
import warnings

from PIL import Image

import base.print_v as p

import display.display_images as di

debug_keys = ["w"]
debug_show_small_subsets = True  # show the subsets per direction (normal and up-side down)
debug_show_small_subsets_polygons = True
debug_show_subsets_polygons_unmerged = True
debug_show_subset_polygons = True  # show the found text-polygons per subset
debug_show_all_polygons = True
debug_print_raw_data_all = True
debug_print_raw_data_final = True


def extract_text_tesseract(img, image_id, binarize_image=None,
                           subset_height=None, nr_small_subsets=None,
                           min_confidence=None, min_area=None,
                           padding_x=None, padding_y=None,
                           catch=True, verbose=False, pbar=None):
    """
    extract_text_tesseract(img, image_id, return_bounds, binarize_image, min_confidence,
                 min_area, padding, catch, verbose, pbar):
    This function is looking for text on the image and trying to recognize it (ocr). Note that
    this function is only looking at the edges of the images and not for text in the center of
    the image. Additionally, it can return bounding boxes for where the text is located.
    Args:
        img (np-array): The image for in which we are looking for text
        image_id (String): The image-image_id of the image in which we are looking for text
        binarize_image (Bool): If true, the image will be converted into a binary image
            before text is extract. Usually this improves the text extraction.
        min_confidence (Integer): The minimum confidence for a recognized text. 0 means no
            confidence, 100 means absolute confidence.
        min_area (Integer): The minimum area in pixels, that the polygon around a recognized
            text must have. This helps to remove small text-boxes with only one letter.
        padding_x (Integer): How many pixels will be added at every horizontal side of the
            polygon of a recognized text
        padding_y (Integer): Same as padding_x, but for the vertical side
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        text_content:
        text_bounds:
        text_conf:
    """

    p.print_v(f"Start: extract_text_tesseract ({image_id})", verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the subset height
    if subset_height is None:  # how high is the subset?
        subset_height = json_data["extract_text_subset_height"]

    if nr_small_subsets is None:
        nr_small_subsets = json_data["extract_text_nr_small_subsets"]

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
        subset = copy.deepcopy(subset_orig)

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
            small_subset = subset[:, small_subset_left:small_subset_right]

            # as text can be written upside-down (they really didn't handle these images good...)
            # we want to look at every subset twice: normal and upside-down

            # first the subset we are not rotating, then the rotated one
            small_subset_1 = copy.deepcopy(small_subset)
            small_subset_2 = np.rot90(copy.deepcopy(small_subset), 2)

            if debug_show_small_subsets and key in debug_keys:
                di.display_images([small_subset_1, small_subset_2],
                                  title=f"Subsets ({i+1}/{nr_small_subsets}) for '{key}'")

            # here we save the boxes and the text we found
            text_pos_rel = []
            text_pos_abs = []
            text_content = []
            text_conf = []

            # convert to opencv image
            pil_1 = Image.fromarray(small_subset_1)
            pil_2 = Image.fromarray(small_subset_2)

            # find text in image
            small_data_1 = pytesseract.image_to_data(pil_1, lang='eng',
                                                     output_type='data.frame',
                                                     config="-c page_separator='' ")
            small_data_2 = pytesseract.image_to_data(pil_2, lang='eng',
                                                     output_type='data.frame',
                                                     config="-c page_separator='' ")

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

        # merge data based on position
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            lines = data.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'] \
                .apply(lambda x: ' '.join(list(x))).tolist()
            confs = data.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['conf'].mean().tolist()
            left = data.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['left'].min().tolist()
            right = data.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['right'].max().tolist()
            top = data.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['top'].min().tolist()
            bottom = data.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['bottom'].max().tolist()

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

            if key == "w":
                print(min_area, poly_rel.area)

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
            text_pos_rel.append(poly_rel)
            text_pos_abs.append(poly_abs)
            text_content.append(lines[i])
            text_conf.append(confs[i])

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
    text_content_all = [item for sublist in text_content_all for item in sublist]
    text_position_all = [item for sublist in text_position_all for item in sublist]
    text_conf_all = [item for sublist in text_conf_all for item in sublist]

    p.print_v(f"Finished: extract_text_tesseract ({image_id})", verbose, pbar=pbar)

    return text_content_all, text_position_all, text_conf_all


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
    image_id = "CA196631L0061"

    import base.load_image_from_file as liff

    _img = liff.load_image_from_file(image_id)

    # di.display_images(_img)

    text, text_pos, text_conf = extract_text_tesseract(_img, image_id, binarize_image=True, verbose=True)
    print(text)
    print(text_pos)
    print(text_conf)
