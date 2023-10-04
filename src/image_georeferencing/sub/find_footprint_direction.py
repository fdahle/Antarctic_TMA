import numpy as np

import display.display_images as di

debug_show_points = False


def find_footprint_direction(img, points, step_size=2500):
    """
    find_footprint_direction(img, points):
    This function looks at the position of tie-points in an image and then returns step_x, step_y to
    put the tie-points as close to the middle as possible. The image is divided in 4 quadrants:
        # 1 2
        # 3 4
    For every quadrant the number of points is counted. The algorithm tries to estimate x and y,
    so that the number of points in each quadrant is more similar.
    Args:
        img (np-array): The image for which we want to find the tie-point direction (Just used for shape)
        points (np.array): The tie-points (x,y) we have for this image
        step_size (int, 2500): Integer that states how much we want to shift a direction. As bigger, as
            higher the shift.
    Returns:
        step_y (int): How many pixels in y should the image be shifted to centralize the tie-points
        step_x (int): How many pixels in x should the image be shifted to centralize the tie-points
    """

    assert isinstance(step_size, int), "The step size must be an integer"

    # first get the breakpoints for the image quadrants
    if len(img.shape) == 2:
        image_height = img.shape[0]
        image_width = img.shape[1]

    elif len(img.shape) == 3:
        if img.shape[0] == 3:
            image_height = img.shape[1]
            image_width = img.shape[2]
        elif img.shape[2] == 3:
            image_height = img.shape[0]
            image_width = img.shape[1]
        else:
            raise ValueError(f"Image shape {img.shape} not supported")
    else:
        raise ValueError(f"Image shape {img.shape} not supported")

    mid_x = int(image_width / 2)
    mid_y = int(image_height / 2)

    # now get the points per quadrant
    idx_q_1 = np.where((points[:, 0] <= mid_x) & (points[:, 1] <= mid_y))[0]
    idx_q_2 = np.where((points[:, 0] > mid_x) & (points[:, 1] <= mid_y))[0]
    idx_q_3 = np.where((points[:, 0] <= mid_x) & (points[:, 1] > mid_y))[0]
    idx_q_4 = np.where((points[:, 0] > mid_x) & (points[:, 1] > mid_y))[0]

    # how much percentage of points are in each quadrant
    perc_q_1 = round(len(idx_q_1) / points.shape[0], 2)
    perc_q_2 = round(len(idx_q_2) / points.shape[0], 2)
    perc_q_3 = round(len(idx_q_3) / points.shape[0], 2)
    perc_q_4 = round(len(idx_q_4) / points.shape[0], 2)

    # how big of a step we need to do
    step_left = np.average([perc_q_2, perc_q_4]) * step_size
    step_right = np.average([perc_q_1, perc_q_3]) * step_size
    step_bottom = np.average([perc_q_1, perc_q_2]) * step_size
    step_top = np.average([perc_q_3, perc_q_4]) * step_size

    # print("l, r, b, t")
    # print(step_left, step_right, step_bottom, step_top)

    if debug_show_points:
        # Line from top of image to bottom of image at mid_x
        vertical_line = (mid_x, 0, mid_x, image_height)

        # Line from left of image to right of image at mid_y
        horizontal_line = (0, mid_y, image_width, mid_y)

        di.display_images([img], lines=[vertical_line, horizontal_line], points=[points])

    step_y = int(step_bottom - step_top)
    step_x = int(step_left - step_right)

    # print("B", step_bottom, "T", step_top, "Y", step_y)
    # print("L", step_left, "R", step_right, "X", step_x)

    # TODO: IDEAS
    # - calculate distance from point to center
    # - also rotate the cross ?

    return step_y, step_x


if __name__ == "__main__":
    _points = np.asarray([
        [473, 492],
        [556, 268],
        [558, 251],
        [548, 335],
        [562, 251],
        [566, 245],
        [566, 249],
        [565, 312],
        [566, 231],
        [564, 256],
        [576, 352],
        [576, 352],
        [573, 274],
        [573, 274],
        [592, 291],
        [592, 291],
        [607, 297],
        [591, 343],
        [615, 311],
        [640, 405],
        [652, 327],
        [652, 310],
        [669, 336],
        [632, 573]])
    _conf = [0.32276562, 0.3870658, 0.47237828, 0.4701222, 0.38276365, 0.33112985,
             0.31735426, 0.37993783, 0.23197573, 0.2287998, 0.51280385, 0.44722897,
             0.43692392, 0.36414182, 0.51947737, 0.32001877, 0.45340052, 0.21514753,
             0.74041903, 0.34701747, 0.33185983, 0.47278568, 0.32856208, 0.2329101]

    _img = np.zeros((472 * 2, 456 * 2))

    _step_y, _step_x = find_footprint_direction(_img, _points)
    print(_step_y, _step_x)
