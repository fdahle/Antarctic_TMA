import numpy as np

from typing import Optional, Dict, Tuple, List


def create_mask(image: np.ndarray,
                fid_marks: Optional[Dict[str, Tuple[int, int]]] = None,
                ignore_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
                default_fid_position: int = 500,
                min_border_width: int = None) -> np.ndarray:
    """
    This function generates a mask for an image based on provided fiducial marks (everything outside
    these marks is masked). If no marks are provided, default positions are used. Additionally,
    specific boxes within the image can be ignored by setting them to zero in the mask.
    Masked areas are set to 0, unmasked areas are set to 1.

    3 ## 7 ## 2
    #         #
    5 # PPA # 6
    #         #
    1 ## 8 ## 4

    Args:
        image: A numpy array representing the image to mask.
        fid_marks: An optional dictionary of fiducial marks with keys as string identifiers
                   and values as tuples representing positions (x, y). If None, default positions are used.
        ignore_boxes: An optional list of boxes to ignore, each specified as a tuple of four integers
                      (x1, y1, x2, y2) representing the top left (x1, y1) and bottom right (x2, y2) corners.
        default_fid_position: An optional integer defining the default position for fiducial marks
                              if none are provided. Defaults to 500.
        min_border_width: An optional integer defining the minimum border width to apply to the mask.
    Returns:
        A numpy array representing the masked image, where regions outside the specified fiducial
        marks and ignore boxes are set to zero.
    """

    # create base mask
    mask = np.ones_like(image)

    # Define default positions
    default_positions = {
        "1": (default_fid_position, mask.shape[0] - default_fid_position),
        "2": (mask.shape[1] - default_fid_position, default_fid_position),
        "3": (default_fid_position, default_fid_position),
        "4": (mask.shape[1] - default_fid_position, mask.shape[0] - default_fid_position),
    }

    # Initialize fid_marks if None, or fill in missing/default for existing keys
    if fid_marks is None:
        fid_marks = default_positions
    else:
        for key, default_position in default_positions.items():
            # If key doesn't exist in fid_marks or value is None, set to default position
            fid_marks[key] = fid_marks.get(key, default_position)

    # get the min and max x/y values from the fid marks
    min_x = max(fid_marks["3"][0], fid_marks["1"][0])
    max_x = min(fid_marks["2"][0], fid_marks["4"][0])
    min_y = max(fid_marks["3"][1], fid_marks["2"][1])
    max_y = min(fid_marks["1"][1], fid_marks["4"][1])

    # Apply the min_border_width if specified
    if min_border_width is not None:
        min_x = max(min_x, min_border_width)
        max_x = min(max_x, mask.shape[1] - min_border_width)
        min_y = max(min_y, min_border_width)
        max_y = min(max_y, mask.shape[0] - min_border_width)

    # mask the borders
    # Set top and bottom regions to 0
    mask[:min_y, :] = 0
    mask[max_y:, :] = 0

    # Set left and right regions to 0
    mask[:, :min_x] = 0
    mask[:, max_x:] = 0

    # Mask the ignore_boxes if any
    if ignore_boxes is not None:
        for box in ignore_boxes:
            x1, y1, x2, y2 = box
            mask[int(y1):int(y2), int(x1):int(x2)] = 0

    return mask
