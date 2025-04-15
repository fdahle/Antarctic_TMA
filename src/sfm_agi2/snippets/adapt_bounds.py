import numpy as np

def adapt_bounds(bounds, pixel_size):

    # get min and max values
    min_x = bounds[0]
    min_y = bounds[1]
    max_x = bounds[2]
    max_y = bounds[3]

    # round values to the next step of pixel size
    new_min_x = np.floor(min_x / pixel_size) * pixel_size
    new_min_y = np.floor(min_y / pixel_size) * pixel_size
    new_max_x = np.ceil(max_x / pixel_size) * pixel_size
    new_max_y = np.ceil(max_y / pixel_size) * pixel_size

    # get differences between old and new bounds
    diff_min_x = new_min_x - min_x
    diff_min_y = new_min_y - min_y
    diff_max_x = new_max_x - max_x
    diff_max_y = new_max_y - max_y

    print(f"[INFO] Bounds adapted with ({diff_min_x}, {diff_min_y}, {diff_max_x}, {diff_max_y}")
    print(f"[INFO] New bounds: ({new_min_x}, {new_min_y}, {new_max_x}, {new_max_y})")

    # return new bounds
    return (new_min_x, new_min_y, new_max_x, new_max_y)