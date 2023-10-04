import copy
import cv2
import numpy as np

import base.print_v as p


def resize_image(input_img, new_size, size="size", interpolation="nearest",
                 catch=True, verbose=False, pbar=None):

    """
    resize_image(input_img, height, width, interpolation, catch, verbose, pbar):
    This function resizes an image to a specific size
    Args:
        input_img (np-array): The numpy array containing the image
        new_size (tuple): The new size of the image.
            If size="size", it is (height, width),
            If size="proportion" it is (height_factor, width_factor)
        size (string): Describes the type of the new_size tuple. Can be "size" or "proportion"
        interpolation (string): How should the image be resized
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        img (np-array): The resized image
    """

    p.print_v("Start: resize_image", verbose=verbose, pbar=pbar)

    # get the new height and width
    if size == "size":
        height = new_size[0]
        width = new_size[1]
    elif size == "proportion":
        if len(new_size) == 2:
            height = int(input_img.shape[0] * new_size[0])
            width = int(input_img.shape[1] * new_size[1])
        else:
            height = int(input_img.shape[0] * new_size)
            width = int(input_img.shape[1] * new_size)
    else:
        raise ValueError("variable 'size' is not defined")

    # copy to not change original
    img = copy.deepcopy(input_img)

    try:
        # check if we need to move the axis for cv2 resize
        bool_axis_moved = False
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = np.moveaxis(img, 0, 2)
            bool_axis_moved = True

        # the actual resizing
        if interpolation == "nearest":
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)  # noqa
        else:
            raise NotImplementedError("Other methods of resizing not implemented yet")

        # move axis back
        if bool_axis_moved:
            img = np.moveaxis(img, 2, 0)
    except (Exception,) as e:
        if catch:
            raise e
        else:
            p.print_v("Failed: resize_image", verbose=verbose, pbar=pbar)
            return None

    p.print_v("Finished: resize_image", verbose=verbose, pbar=pbar)

    return img
