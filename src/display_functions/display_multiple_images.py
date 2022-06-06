import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

import create_cmap as cm
import get_good_squares as ggs

"""
This function can be used to display multiple images in one figure with some additional settings
Input:
- list_of_images: A list of images that should be displayed (usually a np-array)
- list_of types: A list of strings specifiying the type of the images (changes how an image is displayed).
        Possible values are ["auto", "color", "gray", "segmented", "prob"]. If the value 'auto'
        is selected, the program tries to determine the type itself
- title: Optional title for the figure
- subtitles: Optional subtitles per image (again a list)
- bboxe: Optional boxes on the image (again a list)
- lines: Optional lines for the images (again a list)
"""

def display_multiple_images(list_of_images, list_of_types=None, title=None, subtitles=None, bboxes=None, lines=None):

    assert isinstance(list_of_images, list), "Please enter the input data as a list"
    assert len(list_of_images) >= 1

    if list_of_types is not None:
        assert len(list_of_images) == len(list_of_types)
    if subtitles is not None:
        assert len(list_of_images) == len(subtitles)
    if bboxes is not None:
        assert len(list_of_images) == len(bboxes)

    def get_img_type(input_img):

        # check for 0 size and then stop checking -> otherwise error
        for elem in input_img.shape:
            if elem == 0:
                return "binary"

        if input_img is None:
            _img_type = None

        elif len(input_img) == 0:
            _img_type = None

        elif len(input_img.shape) == 2:

            if np.nanmax(input_img) <= 1:
                _img_type = "binary"
            elif np.nanmax(input_img) > 8:
                _img_type = "gray"
            else:
                _img_type = "segmented"

        elif len(input_img.shape) == 3:
            if input_img.shape[2] == 3:
                _img_type = "color"
            else:
                _img_type = "undefined"
        else:
            _img_type = "undefined"

        print(_img_type)

        return _img_type

    len_lst = len(list_of_images)
    root = math.sqrt(len_lst)

    # define the shape of the subplots
    if len_lst == 1:
        f, axarr = plt.subplots(1)

        len_y, len_x = 1, 1

    # everything under 3 is just fine as a row
    elif len_lst <= 3:
        f, axarr = plt.subplots(1, len_lst)

        len_y, len_x = 1, len_lst

    # square numbers are easy
    elif int(root + 0.5) ** 2 == len_lst:

        sq_num = int(math.sqrt(len_lst))
        f, axarr = plt.subplots(sq_num, sq_num)

        len_y = axarr.shape[0]
        len_x = axarr.shape[1]

    # all the rest
    else:
        positions = ggs.get_good_squares(list_of_images)

        f, axarr = plt.subplots(positions)

        len_y = axarr.shape[0]
        len_x = axarr.shape[1]

    # add the lines to the images
    if lines is not None:
        for i, elem in enumerate(lines):
            if elem is None:
                continue
            cv2.line(list_of_images[i], (elem[0], elem[1]), (elem[2], elem[3]), (255, 255, 0), 2)

    # add title
    if title is not None:
        f.suptitle(title)

    img_counter = 0
    for y in range(len_y):
        for x in range(len_x):

            if img_counter >= len_lst:
                f.delaxes(axarr[y, x])
                continue

            img = list_of_images[img_counter]

            if list_of_types is None:
                img_type = get_img_type(img)
            else:
                img_type = list_of_types[img_counter]

            if img_type is None:
                img = np.zeros((2, 2))

            if len_y > 1:
                if img_type == "gray":
                    axarr[y, x].imshow(img, cmap="gray", interpolation=None, vmin=0, vmax=255)
                elif img_type == "segmented":
                    cmap, norm = cm.create_cmap()
                    axarr[y, x].imshow(img, cmap=cmap, norm=norm, interpolation=None)
                elif img_type == "binary" or img_type == "color":
                    axarr[y, x].imshow(img, interpolation=None)
                if subtitles is not None:
                    axarr[y, x].title.set_text(subtitles[img_counter])
                if bboxes is not None:
                    bbox = bboxes[img_counter]
                    if bbox is not None and len(bbox) > 0:
                        print("VV", bbox)
                        rect = patches.Rectangle((bbox[0], bbox[2]), bbox[1] - bbox[0], bbox[3] - bbox[2],
                                                 linewidth=1, edgecolor='r', facecolor='none')
                        axarr[y, x].add_patch(rect)


            # if the figure contains only one row of images, everything must be shown different
            else:
                if img_type == "gray":
                    axarr[x].imshow(img, cmap="gray", interpolation=None, vmin=0, vmax=255)
                elif img_type == "segmented":
                    cmap, norm = cm.create_cmap()
                    axarr[x].imshow(img, cmap=cmap, norm=norm, interpolation=None)
                elif img_type == "binary" or img_type == "color":
                    axarr[x].imshow(img, interpolation=None)
                if subtitles is not None:
                    axarr[x].title.set_text(subtitles[img_counter])
                if bboxes is not None:
                    bbox = bboxes[img_counter]
                    if bbox is not None and len(bbox) > 0:
                        rect = patches.Rectangle((bbox[0], bbox[2]), bbox[1] - bbox[0], bbox[3] - bbox[2],
                                                 linewidth=1, edgecolor='r', facecolor='none')
                        axarr[x].add_patch(rect)

            img_counter += 1

    plt.show()
