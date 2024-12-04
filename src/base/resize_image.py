"""Resize an image"""

# Library imports
import copy
import cv2
import numpy as np

def resize_image(input_img: np.ndarray,
                 new_size: tuple[int, int] | float, size: str = "size",
                 interpolation: str = "nearest",
                 return_scale_factor: bool = False,
                 verbose: bool = False) -> np.ndarray:
    """
    Resizes an image to a specific size or proportion by interpolating it using methods from OpenCV.

    Args:
        input_img (np.ndarray): The numpy array containing the image.
        new_size (Union[Tuple[int, int], float]): The new size or proportion of the image.
            If `size` is "size", it should be (height, width).
            If `size` is "proportion", it should be a single float value representing both height and width factor.
        size (str): Type of resizing to perform. Options are "size" for specific dimensions or "proportion" for scaling
            by a factor.
        interpolation (str): Method of interpolation for resizing. Currently, only "nearest" is implemented.
        return_scale_factor (bool): If true, returns the scaling factor for x and y.
        verbose (bool): If true, prints status of operations.

    Returns:
        np.ndarray: The resized image.
    """

    # Determine new height and width
    if size == "size":

        # check if new size has 3 dimension
        if len(new_size) == 3:
            # Remove the dimension with the smallest value
            new_size = [dim for dim in new_size if dim != min(new_size)]

            # Ensure new_size has exactly two dimensions
        if len(new_size) != 2:
            raise ValueError("After processing, new_size must have exactly two dimensions.")

        height, width = new_size

        if len(input_img.shape) == 2:
            scale_x = width / input_img.shape[1]
            scale_y = height / input_img.shape[0]
        else:
            scale_x = width / input_img.shape[2]
            scale_y = height / input_img.shape[1]

    elif size == "proportion":
        height = int(input_img.shape[0] * new_size)
        width = int(input_img.shape[1] * new_size)

        scale_x = scale_y = new_size
    else:
        raise ValueError("The 'size' parameter should be either 'size' or 'proportion'.")

    # Copy to not change original image
    img = copy.deepcopy(input_img)

    # Check if axis needs to be moved for cv2 resize
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = np.moveaxis(img, 0, 2)
        bool_axis_moved = True
    else:
        bool_axis_moved = False

    # boolean images are not supported by cv2.resize
    is_bool = False
    if img.dtype == bool:
        is_bool = True
        img = img.astype(np.uint8) * 255  # Convert to 0 and 255 for binary representation

    # Perform the actual resizing
    if interpolation == "nearest":
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        raise NotImplementedError("Only 'nearest' interpolation method is currently implemented.")

    # Move axis back if it was moved before resizing
    if bool_axis_moved:
        img = np.moveaxis(img, 2, 0)

    if verbose:
        print(f"Image resized from {input_img.shape} to {img.shape}")

    # convert back to bool
    if is_bool:
        img = img.astype(bool)

    # Return both image and scaling factors if requested
    if return_scale_factor:
        return img, scale_x, scale_y
    else:
        return img
