import numpy as np
from PIL import Image

def export_thumbnail(image_array: np.ndarray, save_path: str, size: tuple = (500,500)):
    """
    Create a thumbnail from a NumPy array and save it as a JPEG file.

    Parameters:
    - image_array: np.ndarray - The input image as a NumPy array.
    - save_path: str - The path where the thumbnail will be saved.
    - size: tuple - The desired size of the thumbnail, e.g., (width, height).
    """

    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image_array)

    # Resize the image to the specified size
    thumbnail = image.resize(size)

    # Save the thumbnail as a JPEG file
    thumbnail.save(save_path, "JPEG")