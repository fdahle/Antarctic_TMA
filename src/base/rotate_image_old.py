import cv2
import numpy as np


def rotate_image_old(image: np.ndarray,
                 angle: float,
                 expand: bool = True,
                 interpolate: bool = True,
                 return_rot_matrix: bool = False) -> (np.ndarray, np.ndarray | None):
    """
    Rotates an image by a given angle with an automatic check for CHW format.

    The function assumes that if a 3D image's first dimension is less than 10
    and the second dimension is at least 10, the image is in channel-first (CHW) format.
    In that case, it transposes the image to height–width–channel (HWC) for processing
    and converts the result back to CHW before returning.

    Args:
        image (np.ndarray): Input image. Supports grayscale (2D) or color images.
                            Color images can be in CHW (channels, height, width) or
                            HWC (height, width, channels) format.
        angle (float): Rotation angle in degrees. Positive values rotate counter-clockwise.
        expand (bool): If True, the output image dimensions are expanded to fit the entire rotated image.
        interpolate (bool): If True, uses bilinear interpolation; otherwise uses nearest-neighbor.
        return_rot_matrix (bool): If True, returns the 2x3 rotation matrix along with the rotated image.

    Returns:
        rotated_image (np.ndarray): The rotated image, in the same format as the input.
        rotation_matrix (np.ndarray, optional): The 2x3 affine rotation matrix (if return_rot_matrix is True).
    """

    # Handle bool images: convert to uint8 (0 or 255) for rotation
    is_bool = False
    if image.dtype == bool or image.dtype == np.bool_:
        is_bool = True
        image = (image.astype(np.uint8)) * 255


    # Check if image is likely in CHW format (channels, height, width)
    was_chw = False
    if len(image.shape) == 3:
        # Heuristic: first dimension is channels if it is less than 10 and height is at least 10
        if image.shape[0] < 10 and image.shape[1] >= 10:
            was_chw = True
            image = np.transpose(image, (1, 2, 0))  # Convert CHW -> HWC

    # Get image dimensions and center
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # If angle is effectively 0 (or a multiple of 360), no rotation is needed.
    # If angle is effectively 0 (or a multiple of 360), no rotation is needed.
    if angle % 360 == 0:
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        result = image if not was_chw else np.transpose(image, (2, 0, 1))
        if is_bool:
            # Convert back to bool if needed
            result = (result >= 128)
        return (result, identity) if return_rot_matrix else result

    # Set interpolation flag
    interp_flag = cv2.INTER_LINEAR if interpolate else cv2.INTER_NEAREST

    # Compute rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    if expand:
        # Compute new dimensions for the rotated image
        abs_cos = abs(rot_mat[0, 0])
        abs_sin = abs(rot_mat[0, 1])
        new_w = int((h * abs_sin) + (w * abs_cos))
        new_h = int((h * abs_cos) + (w * abs_sin))

        # Adjust the rotation matrix to account for translation
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=interp_flag)
    else:
        rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=interp_flag)

    # If input was in CHW format, convert the result back to CHW.
    if was_chw:
        rotated = np.transpose(rotated, (2, 0, 1))
    # If the original image was boolean, convert the rotated result back to bool.
    if is_bool:
        rotated = (rotated >= 128)

    return (rotated, rot_mat) if return_rot_matrix else rotated
