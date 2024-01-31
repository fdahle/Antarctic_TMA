import copy
import json
import os

import base.print_v as p


def mask_borders(mask, bounds, buffer_val=None,
                 catch=True, verbose=False, pbar=None):
    """
    mask_borders(mask, fid_marks, buffer_val, catch, verbose, pbar):
    This function takes an input mask and adds more masked values based on the borders (based on
    the fid marks). The smaller x-or-y value of the fid-marks are taken, so that we get the biggest
    possible mask. 0 means masked, 1 means not masked
    Args:
        mask (np-array): The binary numpy-array containing the mask
        fid_marks (Pandas DataFrame): A dataframe containing the position of the fid-points
        buffer_val (integer): How many pixels we will add to each border to increase the
            masked area
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        mask (np-array): the new mask with the border information included
    """

    p.print_v("Start: mask_border", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if buffer_val is None:
        buffer_val = json_data["mask_border_buffer_val"]

    # create a deepcopy to not change the original mask
    mask = copy.deepcopy(mask)

    try:
        # add a buffer value
        min_x = int(bounds[0] + buffer_val)
        max_x = int(bounds[1] - buffer_val)
        min_y = int(bounds[2] + buffer_val)
        max_y = int(bounds[3] - buffer_val)

        # set the mask
        mask[:, :min_x] = 0
        mask[:, max_x:] = 0
        mask[:min_y, :] = 0
        mask[max_y:, :] = 0
    except (Exception,) as e:
        if catch:
            p.print_v("Failed: mask_border", verbose=verbose, pbar=pbar)
            return None
        else:
            raise e

    p.print_v("Finished: mask_border", verbose=verbose, pbar=pbar)

    return mask
