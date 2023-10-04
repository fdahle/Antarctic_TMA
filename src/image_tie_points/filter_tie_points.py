import numpy as np

import base.print_v as p

from tqdm import tqdm


# WARNING: Note that there's also a filtering function in 'find_tie_points.py' It is required there to
# speed up the process of finding tie-points

def filter_tie_points(tie_points, conf, mask_1, mask_2,
                      catch=True, verbose=False, pbar=None):
    """
    This function filters detected tie-points based on masks of the two images of the tie-points.
    Where the mask-value is 0, the ti-points is filtered.
    Args:
        tie_points (np-array): The tie-points we want to filter
        conf (np.array): The confidences of the tie-points
        mask_1 (np-array): binary mask for image 1
        mask_2 (np-array): binary mask for image 2
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        tie_points_filtered (np-array): The filtered tie-points
        conf_filtered (np.array): The filtered confidences
    """

    p.print_v("Start: filter_tie_points", verbose=verbose, pbar=pbar)

    try:

        # check that the number of tie-points is similar
        assert tie_points.shape[0] == conf.shape[0]

        # get the single coordinates from data and convert to list
        x_1_list = tie_points[:, 0]
        y_1_list = tie_points[:, 1]
        x_2_list = tie_points[:, 2]
        y_2_list = tie_points[:, 3]

        # this array will contain all points that are left (aka not in the mask)
        tie_points_filtered = []
        conf_filtered = []

        # iterate through all the points
        for idx in range(tie_points.shape[0]):

            x_1 = x_1_list[idx]
            y_1 = y_1_list[idx]
            x_2 = x_2_list[idx]
            y_2 = y_2_list[idx]
            quality = conf[idx]

            # check if one of the points is masked
            if mask_1[int(y_1), int(x_1)] == 0 or mask_2[int(y_2), int(x_2)] == 0:
                continue

            # save the other points
            tie_points_filtered.append([x_1, y_1, x_2, y_2])
            conf_filtered.append(round(float(quality), 3))

        # convert list to array
        tie_points_filtered = np.array(tie_points_filtered)

    except (Exception,) as e:
        if catch:
            p.print_v("Failed: filter_tie_points", verbose=verbose, pbar=pbar)
            return None, None
        else:
            raise e

    p.print_v("Finished: filter_tie_points", verbose=verbose, pbar=pbar)

    return tie_points_filtered, conf_filtered
