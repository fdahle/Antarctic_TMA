import numpy as np
import rasterio.transform

from typing import Tuple

debug_print_values = False


def verify_image_geometry(image: np.ndarray, transform: np.ndarray,
                          length_difference: float = 15.0, pixel_difference: float = 25.0,
                          min_pixel_size: float = 0.1, max_pixel_size: float = 2.0,
                          angle_threshold: float = 10) -> Tuple[bool, str]:
    """
    Verifies the geometric integrity of a geo-referenced image based on specified thresholds for length difference,
    maximum pixel size, and angle threshold.

    Args:
        image (np.ndarray): The image array to verify.
        transform (np.ndarray): Transformation matrix applied to the image.
        length_difference (float, optional): Maximum allowed percentage difference between the sides. Defaults to 15.
        pixel_difference (float, optional): Maximum allowed percentage difference between pixel sizes. Defaults to 25.
        min_pixel_size (float, optional): Minimum allowed pixel size. Defaults to 0.1.
        max_pixel_size (float, optional): Maximum allowed pixel size. Defaults to 2.
        angle_threshold (float, optional): Maximum allowed deviation from 90 degrees for any angle. Defaults to 10.

    Returns:
        bool: Indicates if the image passes verification. True if the image meets all specified criteria,
            False otherwise.
        str: A message string providing feedback on a specific verification check.
            If the image is verified successfully, The string is empty
    """

    # Get the pixel coordinates of the four corners of the image
    rows, cols = image.shape[0], image.shape[1]
    corners = np.array([(0, 0), (0, rows), (cols, rows), (cols, 0)])

    if transform.shape[0] == 3:
        transform = transform.flatten()

    # convert np-array to rasterio transform
    t_transform = rasterio.transform.Affine(*transform)

    # Convert to abs coordinates
    corners_abs = np.array([t_transform * xy for xy in corners])

    # Calculate distances between consecutive corner points
    distances = [np.linalg.norm(corners_abs[(i + 1) % 4] - corners_abs[i]) for i in range(4)]

    # Calculate angles between sides
    angles = []
    for i in range(4):
        # Adjust indexing to access the points directly
        p1 = corners_abs[i]
        p2 = corners_abs[(i + 1) % 4]
        p3 = corners_abs[(i + 2) % 4]

        # Calculate vectors between the points
        v1 = p2 - p1
        v2 = p3 - p2

        # Calculate the angle between the vectors
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        angles.append(angle)

    # calculate with angles are exceeding the threshold
    wrong_angles = [angle for angle in angles if np.abs(angle - 90) > angle_threshold]

    # get difference between height and width of image
    diff = np.abs(((distances[0] - distances[1]) / distances[1]) * 100)

    # get difference between pixel height and width in percent
    pix_diff = np.abs(((np.abs(transform[0]) - np.abs(transform[4])) / np.abs(transform[4])) * 100)

    if debug_print_values:
        print(f"length difference: {diff}")
        print(f"pixel size difference: {pix_diff}")
        print(f"pixel size: {np.abs(transform[0])}, {np.abs(transform[4])}")
        print(f"angles: {angles}")

    # Fail reason 1: Difference between height and width of the image is too big
    if diff > length_difference:
        return False, f"length:{round(diff, 2)}"

    # Fail reason 2: Difference between pixel height and width is too big
    elif pix_diff > pixel_difference:
        return False, f"pixel_diff:{round(pix_diff, 2)}"

    # Fail reason 3: The pixel-size is too big or too small
    elif np.abs(transform[0]) > max_pixel_size or np.abs(transform[4]) > max_pixel_size:
        pix_x = round(transform[0], 4)
        pix_y = round(transform[4], 4)
        return False, f"pixel_size:{pix_x},{pix_y}"

    # Fail reason 4: The image is not a rectangle
    elif wrong_angles:
        wrong_angle_str = ",".join(f"{angle:.2f}" for angle in wrong_angles)
        return False, f"angle:{wrong_angle_str}"

    # Success: The image is valid
    else:
        return True, ""
