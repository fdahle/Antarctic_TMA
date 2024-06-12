"""get z values of gcps"""

# Library imports
import numpy as np

# Local imports
import src.load.load_rema as lr


def get_gcp_height(gcps: np.ndarray, zoom_level: int = 32) -> np.ndarray:
    """
    Retrieves the heights of given ground control points (GCPs) using REMA data.

    Args:
        gcps (np.ndarray): 2D numpy array of GCP coordinates, where each row represents a GCP as (x, y).
        zoom_level (int, optional): Zoom level for loading REMA data. Defaults to 32.

    Returns:
        np.ndarray: 1D numpy array of heights corresponding to the input GCPs.
    """

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
