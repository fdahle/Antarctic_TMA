import numpy as np

from typing import Tuple


def verify_image(image: np.ndarray, transform: np.ndarray, length_difference: float = 15,
                 max_pixel_size: float = 2, angle_threshold: float = 10) -> Tuple[bool, str]:
    """
    Verifies the geometric integrity of a geo-referenced image based on specified thresholds for length difference,
    maximum pixel size, and angle threshold.

    Args:
        image (np.ndarray): The image array to verify.
        transform (np.ndarray): Transformation matrix applied to the image.
        length_difference (float, optional): Maximum allowed percentage difference between the sides. Defaults to 15.
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

    # Convert to abs coordinates
    corners_abs = [transform * xy for xy in corners]

    # Calculate distances between corner points to get all side lengths
    distances = np.linalg.norm(np.diff(corners_abs[:, :4], axis=1), axis=0)

    # Calculate angles between sides
    angles = []
    for i in range(4):
        p1, p2, p3 = corners_abs[:, i], corners_abs[:, (i + 1) % 4], corners_abs[:, (i + 2) % 4]
        v1, v2 = p2 - p1, p3 - p2
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        angles.append(angle)

    # calculate with angles are exceeding the threshold
    wrong_angles = [angle for angle in angles if np.abs(angle - 90) > angle_threshold]

    # get difference between height and width of image
    diff = np.abs(((distances[0] - distances[1]) / distances[1]) * 100)

    # Fail reason 1: Difference between height and width of the image is too big
    if diff > length_difference:
        return False, f"length: {round(diff, 2)}"

    # Fail reason 2: The pixel-size is too big
    elif np.abs(transform[0]) > max_pixel_size or np.abs(transform[4]) > max_pixel_size:
        pix_x = round(transform[0], 4)
        pix_y = round(transform[4], 4)
        return False, f"pixel size: {pix_x},{pix_y}"

    # Fail reason 3: The image is not a rectangle
    elif wrong_angles:
        wrong_angle_str = ", ".join(f"{angle:.2f}" for angle in wrong_angles)
        return False, f"angle : {wrong_angle_str}"

    # Success: The image is valid
    else:
        return True, ""
