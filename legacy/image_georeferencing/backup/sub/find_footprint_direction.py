import numpy as np


def find_footprint_direction(img, points, conf):
    # quadrants are as follows
    # 1 2
    # 3 4

    step_size = 2500

    # first get the breakpoints for the image quadrants
    mid_y = int(img.shape[0] / 2)
    mid_x = int(img.shape[1] / 2)

    # now get the points per quadrant
    idx_q_1 = np.where((points[:, 0] <= mid_x) & (points[:, 1] <= mid_y))[0]
    idx_q_2 = np.where((points[:, 0] > mid_x) & (points[:, 1] <= mid_y))[0]
    idx_q_3 = np.where((points[:, 0] <= mid_x) & (points[:, 1] > mid_y))[0]
    idx_q_4 = np.where((points[:, 0] > mid_x) & (points[:, 1] > mid_y))[0]

    # get the average conf per quadrant
    if len(idx_q_1) == 0:
        conf_q_1 = 0
    else:
        conf_q_1 = np.average(conf[idx_q_1])
    if len(idx_q_2) == 0:
        conf_q_2 = 0
    else:
        conf_q_2 = np.average(conf[idx_q_2])
    if len(idx_q_3) == 0:
        conf_q_3 = 0
    else:
        conf_q_3 = np.average(conf[idx_q_3])
    if len(idx_q_4) == 0:
        conf_q_4 = 0
    else:
        conf_q_4 = np.average(conf[idx_q_4])

    # TODO: WHAT DO WITH CONF

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

    #print("l, r, b, t")
    #print(step_left, step_right, step_bottom, step_top)

    step_y = int(-step_bottom + step_top)
    step_x = int(-step_left + step_right)

    # TODO: IDEAS
    # - calculate distance from point to center
    # - also rotate the cross ?

    #print("Step y, step x")
    #print(step_y, step_x)

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

    _img = np.zeros((472*2, 456*2))

    find_footprint_direction(_img, _points, _conf)