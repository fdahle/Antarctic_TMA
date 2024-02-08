import numpy as np


def create_mask(image, fid_marks=None, ignore_boxes=None, default_fid_position=500):

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
            top_left, bottom_right = box
            mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 0

    return mask