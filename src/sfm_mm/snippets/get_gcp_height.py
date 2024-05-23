import numpy as np

import src.load.load_rema as lr


def get_gcp_height(gcps, zoom_level=32):
    # get bounding box from gcps
    min_x = np.amin(gcps[:, 0])
    max_x = np.amax(gcps[:, 0])
    min_y = np.amin(gcps[:, 1])
    max_y = np.amax(gcps[:, 1])
    bounding_box = [min_x, min_y, max_x, max_y]

    # load rema data
    rema = lr.load_rema(bounding_box, zoom_level=zoom_level)

    # all heights are stored in this list
    heights = []

    # iterate all gcps
    for gcp in gcps:

        # get relative coordinates
        x = int((gcp[0] - min_x)/zoom_level)
        y = int((gcp[1] - min_y)/zoom_level)

        # get the height and append to list
        height = rema[y, x]
        heights.append(height)

    return np.array(heights)
