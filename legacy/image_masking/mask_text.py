import copy

import base.print_v as p


# textbox is [min_x_abs, min_y_abs, max_x_abs, max_y_abs]

def mask_text(mask, text_string,
              catch=True, verbose=False, pbar=None):
    """
    mask_text(mask, text_position, catch, verbose, pbar):
    This function takes an input mask and adds more masked based on textboxes.
    0 means masked, 1 means not masked
    Args:
        mask (np-array): The binary numpy-array containing the mask
        text_string (string): A string  the position of text in the image
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        mask (np-array): the new mask with the text-boxes included
    """

    p.print_v("Start: mask_text", verbose=verbose, pbar=pbar)

    # copy mask to not change original mask
    mask = copy.deepcopy(mask)

    try:
        # split the string in lists
        text_positions = text_string.split(";")

        # add the position of the text boxes to the images
        for elem in text_positions:
            elem = elem[1:-1]
            bbox = elem.split(",")
            mask[int(float(bbox[1])):int(float(bbox[3])),
                 int(float(bbox[0])):int(float(bbox[2]))] = 0

    except (Exception,) as e:
        if catch:
            p.print_v("Failed: mask_text", verbose=verbose, pbar=pbar)
            return None
        else:
            raise e

    p.print_v("Finished: mask_text", verbose=verbose, pbar=pbar)

    return mask
