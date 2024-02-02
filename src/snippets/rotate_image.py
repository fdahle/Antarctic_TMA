import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle: float,
                 expand: bool = True) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Rotates an image by a given angle, optionally expanding the image to fit the rotated result.

    Args:
        image (np.ndarray): The image to rotate.
        angle (float): The rotation angle in degrees. Positive values mean counter-clockwise rotation.
        expand (bool, optional): Whether to expand the output image to fit the rotated image. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray, tuple[int, int]]: A tuple containing the rotated image, the rotation matrix used
            for the transformation, and the center of the original image.
    """
    # calculate image center
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    if expand:
        # Calculate the new dimensions to accommodate the rotated image
        cos_theta = abs(rotation_matrix[0, 0])
        sin_theta = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_theta) + (width * cos_theta))
        new_height = int((height * cos_theta) + (width * sin_theta))

        # Adjust the rotation matrix to account for translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Apply the rotation matrix to the image with new dimensions
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    else:
        # Apply the rotation matrix to the image with original dimensions
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image, rotation_matrix, center
