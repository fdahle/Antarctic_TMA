import src.base.resize_image as ri

def resize_max(arr, max_size):
    """
    Resize a NumPy array so the largest dimension is at most `max_size`, preserving aspect ratio.

    Parameters:
        arr (numpy.ndarray): Input array with shape (y, x).
        max_size (int): Maximum size for the largest dimension.

    Returns:
        numpy.ndarray: Rescaled array.
        tuple: Scale factors for x and y.
    """
    y, x = arr.shape
    # Calculate the scaling factor
    scale = max_size / max(y, x)
    if scale >= 1.0:  # No need to resize if already within bounds
        return arr, 1.0, 1.0

    # Calculate new dimensions
    new_y = int(y * scale)
    new_x = int(x * scale)

    # Rescale the array using numpy
    resized_arr = ri.resize_image(arr, (new_y, new_x), size="size", interpolation="nearest")

    return resized_arr, scale, scale
