import cv2
import pyvips
import numpy as np


def rotate_image(image: np.ndarray,
                 angle: float,
                 expand: bool = True,
                 interpolate: bool = True,
                 force_pyvips: bool = False,
                 return_rot_matrix: bool = False) -> (np.ndarray, np.ndarray | None):
    """
    Rotates an image by a given angle. For large images (w or h > 20000px),
    uses pyvips for memory-efficient rotation. Otherwise uses OpenCV warpAffine.

    The function also automatically checks for CHW format (if channels < 10 and height >= 10)
    and converts to HWC for processing. Boolean images are converted to [0, 255].

    Args:
        image (np.ndarray): Input image (2D or 3D).
                            - For color images, can be CHW (channels, height, width) or
                              HWC (height, width, channels).
                            - For boolean, values are converted to [0, 255].
        angle (float): Rotation angle in degrees (counter-clockwise).
        expand (bool): If True, the output image is expanded to fit the entire rotated image.
        interpolate (bool): If True, uses bilinear interpolation; otherwise nearest-neighbor.
        return_rot_matrix (bool): If True, also return the 2×3 rotation matrix (for the OpenCV path only).

    Returns:
        rotated_image (np.ndarray): The rotated image, same shape format as input.
        rotation_matrix (np.ndarray, optional): 2×3 affine matrix (only if return_rot_matrix is True
                                                and OpenCV was used).
    """
    # 1) Preprocessing: handle bool, handle CHW->HWC
    is_bool = False
    if image.dtype == bool or image.dtype == np.bool_:
        is_bool = True
        image = image.astype(np.uint8) * 255

    was_chw = False
    if len(image.shape) == 3:
        if image.shape[0] < 10 and image.shape[1] >= 10:  # channels, height, width
            was_chw = True
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC

    h, w = image.shape[:2]
    c = 1 if image.ndim == 2 else image.shape[2]

    # If angle is multiple of 360, just return original
    if angle % 360 == 0:
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        if was_chw:
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        if is_bool:
            image = (image >= 128)
        return (image, identity) if return_rot_matrix else image

    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    if expand:
        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)
        # shift the center so the image is fully visible
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
    else:
        new_w, new_h = w, h

    # 2) Decide if we use pyvips or OpenCV
    use_pyvips = (w > 20000 or h > 20000) or force_pyvips

    if not use_pyvips:

        # === Use OpenCV ===
        interp_flag = cv2.INTER_LINEAR if interpolate else cv2.INTER_NEAREST

        rotated = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=interp_flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)  # or (0, 0, 0) if you prefer black
        )

        if was_chw:
            rotated = np.transpose(rotated, (2, 0, 1))
        if is_bool:
            rotated = (rotated >= 128)

        return (rotated, M) if return_rot_matrix else rotated

    else:

        # Create the interpolator for rotate().
        interp_obj = pyvips.Interpolate.new("bilinear") if interpolate else pyvips.Interpolate.new("nearest")

        # Convert the numpy image to a pyvips image.
        if c == 1:
            vips_img = pyvips.Image.new_from_memory(
                image.tobytes(), w, h, 1, _numpy2vips_format(image.dtype)
            )
        else:
            vips_img = pyvips.Image.new_from_memory(
                image.tobytes(), w, h, c, _numpy2vips_format(image.dtype)
            )

        A = [M[0, 0], M[0, 1], M[1, 0], M[1, 1]]

        idx = -center[0]
        idy = -center[1]

        odx = center[0]
        ody = center[1]

        if expand:
            # Adjust the output offset to center the expanded canvas as OpenCV does:
            odx += (new_w / 2) - center[0]
            ody += (new_h / 2) - center[1]

        # Step 2: Rotate the image around (0,0)
        rotated_vips = vips_img.affine(A, idx=idx, idy=idy, odx=odx, ody=ody,
                                       interpolate=interp_obj, background=[255] * c,
                                       oarea=[0, 0, new_w, new_h])

        # Ensure the final image has the desired dimensions.
        rotated_vips = rotated_vips.embed(0, 0, new_w, new_h, background=[255] * c)

        # Convert the pyvips image back to a numpy array.
        out_mem = rotated_vips.write_to_memory()
        rotated_arr = np.frombuffer(out_mem, dtype=image.dtype)
        if c == 1:
            rotated_arr = rotated_arr.reshape((new_h, new_w))
        else:
            rotated_arr = rotated_arr.reshape((new_h, new_w, c))
        if was_chw:
            rotated_arr = np.transpose(rotated_arr, (2, 0, 1))
        if is_bool:
            rotated_arr = (rotated_arr >= 128)

        return (rotated_arr, M) if return_rot_matrix else rotated_arr


def _numpy2vips_format(dtype):
    """
    Helper to map NumPy dtypes to pyvips.BandFormat.
    """
    if dtype == np.uint8:
        return 'uchar'
    elif dtype == np.uint16:
        return 'ushort'
    elif dtype == np.int16:
        return 'short'
    elif dtype == np.int32:
        return 'int'
    elif dtype == np.float32:
        return 'float'
    elif dtype == np.float64:
        return 'double'
    else:
        # fallback
        return 'uchar'


if __name__ == "__main__":

    # Define dimensions
    size = 5000
    img = np.full((size, size, 3), 255, dtype=np.uint8)

    th = int(size/200)

    # Draw a blue diagonal line
    cv2.line(img, (0, 0), (size - 1, size - 1), (255, 0, 0), thickness=th)

    # Draw a red border around the image
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (0, 0, 255), thickness=th)

    # Draw green grid lines every 10%
    grid_spacing = int(size / 10)
    for i in range(0, size, grid_spacing):
        cv2.line(img, (0, i), (size - 1, i), (0, 255, 0), thickness=th)
        cv2.line(img, (i, 0), (i, size - 1), (0, 255, 0), thickness=th)

    angle = -173.3
    expand = False

    rotated_img, rot_mat = rotate_image(img, angle, return_rot_matrix=True, expand=expand, force_pyvips=False)
    rotated_img2, rot_mat2 = rotate_image(img, angle, return_rot_matrix=True, expand=expand, force_pyvips=True)


    import src.display.display_images as di
    di.display_images([rotated_img, rotated_img2])