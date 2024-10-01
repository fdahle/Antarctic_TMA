"""Rotate an image"""

# Library imports
import cv2
import numpy as np


def rotate_image(image: np.ndarray,
                 angle: float,
                 expand: bool = True,
                 return_rot_matrix=False) -> (np.ndarray, np.ndarray | None):
    """
    Rotates an image by a given angle, optionally expanding the image to fit the rotated result.

    Args:
        image (np.ndarray): The image to rotate.
        angle (float): The rotation angle in degrees. Positive values indicate a counter-clockwise rotation,
            while negative values indicate a clockwise rotation.
        expand (bool): If True, the output image size is expanded to fit the entire rotated image.
            If False, the output image size is the same as the input, and parts of the rotated image may be cropped.
        return_rot_matrix (bool): If True, the function will return the 2x3 affine rotation matrix used for the
            rotation
    Returns:
        rotated_image (np.ndarray): The rotated image as a NumPy array. The size may change from the original
            based on the angle and whether expansion is requested.
        rotation_matrix (np.ndarray): The 2x3 affine rotation matrix used for the transformation. This matrix can be
            used to understand the rotation and translation applied to the original image.
    """

    # calculate image center
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    if len(image.shape) == 3 and image.shape[0] > 10:
        raise ValueError("Image bands must be in the last dimension")

    # If angle is 0 or a multiple of 360, no need to rotate
    if angle % 360 == 0:
        # Create an identity matrix for the rotation matrix
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0]], dtype=np.float32)
        return image, rotation_matrix

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # init variable
    rotated_image = None

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
        if len(image.shape) == 2:
            rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        else:
            for i in range(image.shape[0]):
                if i == 0:
                    rotated_image = cv2.warpAffine(image[i], rotation_matrix, (new_width, new_height))
                    rotated_image = rotated_image[np.newaxis, ...]
                else:
                    rotated_image = np.vstack((rotated_image, cv2.warpAffine(image[i], rotation_matrix,
                                                                             (new_width, new_height))[np.newaxis, ...]))
    else:
        # Apply the rotation matrix to the image with original dimensions
        if len(image.shape) == 2:
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        else:
            for i in range(image.shape[0]):
                if i == 0:
                    rotated_image = cv2.warpAffine(image[i], rotation_matrix, (width, height))
                    rotated_image = rotated_image[np.newaxis, ...]
                else:
                    rotated_image = np.vstack((rotated_image, cv2.warpAffine(image[i], rotation_matrix,
                                                                             (width, height))[np.newaxis, ...]))

    if return_rot_matrix:
        return rotated_image, rotation_matrix
    else:
        return rotated_image
