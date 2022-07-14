import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import cv2

import create_cmap as cm


def display_single_image(input_img, img_type="auto", title=None, bbox=None, line=None, point=None):

    """
    This function can be used to display a single image with some additional settings
    Args:
    - input_img: The image that should be display (usually a np-array)
    - img_type: A string specifiying the type of the image (changes how an image is displayed).
            Possible values are ["auto", "color", "gray", "segmented", "prob"]. If the value 'auto'
            is selected, the program tries to determine the type itself
    - title: Optional title for the image
    - bbox: Optional box(es) on the image
    - line: Optional line(s) on the image
    - point: Optional point(s) on the image
    Returns:
        
    """


    try:
        _ = input_img.shape
    except (Exception,):
        raise ValueError("The input image is not a valid image")

    input_img = copy.deepcopy(input_img)

    types = ["auto", "color", "gray", "segmented", "prob"]
    if img_type not in types:
        print("The specified display-type is incorrect. Following display-types are allowed:")
        print(types)
        exit()

    fig, ax = plt.subplots()

    # points must be added before the image is plotted
    if point is not None:

        print(point)

        # remove None Values
        point = [i for i in point if i != None]

        if isinstance(point[0], list) is False:
            point = [point]
        for elem in point:
            cv2.circle(input_img, (elem[0], elem[1]), 7, (255, 0, 0), 3)

    # lines must be added before the image is plotted
    if line is not None:
        if isinstance(line, list) is False:
            line = [line]
        for elem in line:
            cv2.line(input_img, (elem[0], elem[1]), (elem[2], elem[3]), (255, 255, 0), 2)

    # try to find the right image type
    if img_type == "auto":

        if len(input_img.shape) == 2:

            if np.amax(input_img) > 8:
                img_type = "gray"
            else:
                img_type = "segmented"

        elif len(input_img.shape) == 3:
            if input_img.shape[2] == 3:
                img_type = "color"
            else:
                img_type = "undefined"
        else:
            img_type = "undefined"

    if img_type == "color":
        ax.imshow(input_img)

    if img_type == "gray":
        ax.imshow(input_img, cmap="gray", vmin=0, vmax=255)

    if img_type == "segmented":
        # create the cmap and norm
        cmap_output, norm_output = cm.create_cmap()
        ax.imshow(input_img, cmap=cmap_output, norm=norm_output, interpolation='none')

    if img_type == "prob":
        cmap_lin = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
        norm_lin = matplotlib.colors.Normalize(vmin=0, vmax=1)
        ax.imshow(input_img, cmap=cmap_lin, norm=norm_lin)

    if img_type == "undefined" or img_type == "auto":
        raise ValueError("The image type could not be extracted. Please specify the image type")

    if bbox is not None:

        # if the first instance is list, then it is a list of boxes
        if isinstance(bbox[0], list) is False:
            # put it in a list so that we can iterate it
            bbox = [bbox]

        for elem in bbox:

            if elem is None or elem[0] is None:
                continue

            rect = patches.Rectangle((elem[0], elem[2]), elem[1] - elem[0], elem[3] - elem[2],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    if title is not None:
        fig.suptitle(title)

    plt.show()
