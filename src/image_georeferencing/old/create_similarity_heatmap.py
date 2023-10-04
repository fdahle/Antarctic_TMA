import json
import numpy as np
import os

import base.print_v as p

def create_similarity_heatmap(big_image, small_image, jump=None):
    """
    create_similarity_heatmap(
    Args:
        big_image:
        small_image:

    Returns:

    """

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the default value for jump
    if jump is None:
        jump = json_data["create_similarity_heatmap_jump"]

    # get width and height from the big image
    big_width = big_image.shape[2]
    big_height = big_image.shape[1]

    # get width and height from the small image
    small_width = small_image.shape[2]
    small_height = small_image.shape[1]



    # initialize the final product -> the heatmap
    heatmap = np.zeros((big_height, big_width))

    # sliding window principle with the small image over the big image
    for y in range(0, big_height - small_height_resized, adapted_step_size_y):
        for x in range(0, big_width - small_width_resized, adapted_step_size_x):
            pass