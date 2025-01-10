"""Extract text from an image with OCR"""

# Library imports
import copy
import cv2 as cv2
import numpy as np
import pandas as pd
import shapely
from numpy import ndarray
from paddleocr import PaddleOCR, draw_ocr  # noqa
from typing import Optional, Any

# Display imports
import src.display.display_images as di

# Constants
USE_GPU = False

# Debug display parameters
debug_keys: list[str] = ["n", "e", "s", "w"]  # for which direction should debug plots/data be shown
debug_show_image: bool = False  # show the complete input image
debug_show_small_subsets: bool = False  # show the initial small subsets
debug_show_small_subsets_polygons: bool = False  # show the polygons on the small subsets
debug_show_subset_polygons: bool = False  # show the polygons on the complete subsets
debug_show_all_polygons: bool = False  # show the polygons on the complete image before merging
debug_show_all_merged_polygons: bool = False  # show the text results on the images with merged polygons already
debug_show_final_polygons: bool = False  # show the final text results on the images

# Debug print parameters
debug_print_raw_final_data: bool = False


def extract_text(
        image: np.ndarray, binarize_image: bool = False,
        subset_height: Optional[int] = None, nr_small_subsets: Optional[int] = None,
        subset_overlap: Optional[int] = None,
        min_confidence: Optional[float] = None, min_area: Optional[int] = None,
        padding_x: Optional[int] = None, padding_y: Optional[int] = None,
        catch: bool = False) -> \
        tuple[Optional[list[str | Any]], Optional[list[Any]], Optional[list[ndarray]]]:
    """
    Extracts text from an image using OCR. The image is divided into four subsets (north, east, south, west) and
    further into smaller subsets. Text is extracted from the smaller subsets and merged to find the final text.
    Args:
        image (np.ndarray): Input image as a numpy array.
        binarize_image (bool): If True, applies binarization to the image before OCR.
        subset_height (Optional[int]): Height of image subsets to process, if None, defaults to 1500.
        nr_small_subsets (Optional[int]): Number of smaller subsets to divide the image into, defaults to 4.
        subset_overlap (Optional[int]): Overlap between subsets, defaults to 200 pixels.
        min_confidence (Optional[float]): Minimum confidence level for OCR to consider text, defaults to 60.
        min_area (Optional[int]): Minimum area for a text bounding box to be considered, defaults to 1000.
        padding_x (Optional[int]): Horizontal padding for text bounding boxes, defaults to 50 pixels.
        padding_y (Optional[int]): Vertical padding for text bounding boxes, defaults to 200 pixels.
        catch (bool): If True, catches and returns None on exception, else raises exception.
    Returns:
        final_text_content: List of extracted text content.
        final_pos: List of bounding box positions for the extracted text.
        final_confs: List of confidence levels for the extracted text.
    """

    # set default values
    subset_height = subset_height or 1500
    nr_small_subsets = nr_small_subsets or 4
    subset_overlap = subset_overlap or 200
    min_confidence = min_confidence or 60
    min_area = min_area or 1000
    padding_x = padding_x or 50
    padding_y = padding_y or 200

    try:

        # init ocr
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False,
                        version='PP-OCR', use_gpu=USE_GPU)

        # init lists for later use
        text_position_all = []
        text_content_all = []
        text_conf_all = []

        # show the complete image
        if debug_show_image:
            di.display_images(image)

        for key in ["n", "e", "s", "w"]:

            # get subset of images based on the direction (makes it easier to find the text)
            if key == "n":
                subset = image[:subset_height, 500:image.shape[1] - 500]
            elif key == "e":
                subset = image[:, image.shape[1] - subset_height:]
            elif key == "s":
                subset = image[image.shape[0] - subset_height:, 500:image.shape[1] - 500]
            elif key == "w":
                subset = image[:, :subset_height]
            else:
                raise ValueError(f"Key '{key}' is not valid")

            subset = copy.deepcopy(subset)

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

                # binarize
                subset[subset < threshold] = 0
                subset[subset != 0] = 1
                subset = 1 - subset

                # apply dilation
                kernel1 = np.ones((3, 3), np.uint8)
                subset = cv2.dilate(subset, kernel1, iterations=1)

                # find contours
                contours, hierarchy = cv2.findContours(subset, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cv2.fillPoly(subset, [cnt], 255)  # noqa

                # set the binarization again
                subset[subset > 0] = 1

                # reverse the binarization
                subset = 1 - subset

            # save width of small subsets (as we divide the subset in smaller parts)
            subset_small_width = int(subset.shape[1] / nr_small_subsets)

            # here we save all data for subset 1 and subset 2
            data_1 = None
            data_2 = None

            # iterate all small subsets
            for i in range(nr_small_subsets):

                # get the position and extent of the small subset
                small_subset_left = i * subset_small_width
                small_subset_right = (i + 1) * subset_small_width

                # add overlap to the subsets
                small_subset_left = small_subset_left - subset_overlap
                small_subset_left = max(0, small_subset_left)
                small_subset_right = small_subset_right - subset_overlap
                small_subset_right = min(subset.shape[1], small_subset_right)

                # get the bounds of the small subset
                small_subset = subset[:, small_subset_left:small_subset_right]

                # as text can be written upside-down (they really didn't handle these images good...)
                # we want to look at every subset twice: normal and upside-down

                # first the subset we are not rotating, then the rotated one
                small_subset_1 = copy.deepcopy(small_subset)
                small_subset_2 = np.rot90(copy.deepcopy(small_subset), 2)

                # show the initial small subsets
                if debug_show_small_subsets and key in debug_keys:
                    style_config = {
                        'title': f"Subsets ({i + 1}/{nr_small_subsets}) for '{key}'"
                    }
                    di.display_images([small_subset_1, small_subset_2], style_config=style_config)

                # here we save the boxes and the text we found
                text_pos_rel = []
                text_pos_abs = []
                text_content = []
                text_conf = []

                # find text in image
                small_data_1 = ocr.ocr(small_subset_1)
                small_data_2 = ocr.ocr(small_subset_2)

                list_1 = []
                if small_data_1[0] is not None:
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
                if small_data_2[0] is not None:
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
                                      bounding_boxes=[boxes1, boxes2])

                # rotate the points back for small data 2
                for index, row in small_data_2.iterrows():
                    left, top = _rot90points(row['left'], row['top'], -2, small_subset_2.shape)
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

            lines = data['text'].tolist()
            confs = data['conf'].tolist()
            left = data['left'].tolist()
            right = data['right'].tolist()
            top = data['top'].tolist()
            bottom = data['bottom'].tolist()

            if debug_print_raw_final_data and key in debug_keys:
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

                # get shape of the subset
                subset_shape = (subset.shape[0], subset.shape[1])

                # rotate points back
                if key == "e":
                    min_x_abs, min_y_abs = _rot90points(min_x_rel, min_y_rel, -1, subset_shape)
                    max_x_abs, max_y_abs = _rot90points(max_x_rel, max_y_rel, -1, subset_shape)
                elif key == "s":
                    min_x_abs, min_y_abs = _rot90points(min_x_rel, min_y_rel, -2, subset_shape)
                    max_x_abs, max_y_abs = _rot90points(max_x_rel, max_y_rel, -2, subset_shape)
                elif key == "w":
                    min_x_abs, min_y_abs = _rot90points(min_x_rel, min_y_rel, -3, subset_shape)
                    max_x_abs, max_y_abs = _rot90points(max_x_rel, max_y_rel, -3, subset_shape)

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
                    min_x_abs = min_x_abs + image.shape[1] - subset_height
                    max_x_abs = max_x_abs + image.shape[1] - subset_height
                elif key == "s":
                    min_x_abs = min_x_abs + 500
                    max_x_abs = max_x_abs + 500
                    min_y_abs = min_y_abs + image.shape[0] - subset_height
                    max_y_abs = max_y_abs + image.shape[0] - subset_height
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
                if max_x_abs > image.shape[1]:
                    max_x_abs = image.shape[1]
                if min_y_abs < 0:
                    min_y_abs = 0
                if max_y_abs > image.shape[0]:
                    max_y_abs = image.shape[0]

                poly_abs = shapely.geometry.box(min_x_abs, min_y_abs, max_x_abs, max_y_abs)

                # save in list
                text_pos_rel.append(poly_rel)  # noqa
                text_pos_abs.append(poly_abs)  # noqa
                text_content.append(lines[i])  # noqa
                text_conf.append(confs[i])  # noqa

            # check if we have found something
            if len(text_pos_rel) > 0:

                if debug_show_subset_polygons and key in debug_keys:
                    di.display_images(subset, polygons=[text_pos_rel])

                # save the stuff we found per subset to a global list
                text_position_all.append(text_pos_abs)
                text_content_all.append(text_content)
                text_conf_all.append(text_conf)

        if debug_show_all_polygons:
            di.display_images(image, polygons=[text_pos_abs])

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
            di.display_images(image, polygons=[merged_text_positions])

        # look for text in the bounding boxes of these merged polygons
        final_text_content = []
        final_pos = []
        final_confs = []
        for merged_poly in merged_text_positions:
            mb = merged_poly.bounds
            merged_box = image[int(mb[1]):int(mb[3]), int(mb[0]):int(mb[2])]
            ocr_results = ocr.ocr(merged_box)
            merged_text = ""
            temp_confs = []
            if ocr_results[0] is not None:
                for elem in ocr_results[0]:
                    merged_text = merged_text + elem[1][0] + " "
                    temp_confs.append(elem[1][1] * 100)
                merged_text = merged_text[:-1]

            # skip empty or otherwise corrupted text
            if len(merged_text) == 0 or np.isnan(np.mean(np.asarray(temp_confs))):
                continue

            final_text_content.append(merged_text)
            final_confs.append(np.mean(np.asarray(temp_confs)))
            final_pos.append(mb)

        if debug_show_final_polygons:
            di.display_images(image, bounding_boxes=[final_pos])

    except (Exception,) as e:
        if catch:
            return None, None, None
        else:
            raise e

    return final_text_content, final_pos, final_confs


# function to rotate points for 90 degrees
def _rot90points(in_x: float, in_y: float, k: int, hw: tuple[int, int]) -> tuple[float, float]:
    """
    Rotates a point (in_x, in_y) 90 degrees counterclockwise k times within a given height and width.

    Args:
        in_x (float): The x-coordinate of the point to rotate.
        in_y (float): The y-coordinate of the point to rotate.
        k (int): The number of 90-degree counterclockwise rotations.
        hw (Tuple[int, int]): A tuple (height, width) representing the dimensions of the space.

    Returns:
        Tuple[float, float]: The new coordinates (x, y) after rotation.

    Raises:
        ValueError: If an invalid value for k is provided.
    """

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


"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract text from an image.")
    parser.add_argument("image_id", help="The id of the image")

    args = parser.parse_args()

    import src.load.load_image as li
    img = li.load_image(args.image_id)

    extract_text(img)
"""
