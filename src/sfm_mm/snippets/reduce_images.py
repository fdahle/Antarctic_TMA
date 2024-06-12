"""remove border from images and masks for MicMac"""

# Library imports
import glob

# Local imports
import src.export.export_tiff as et
import src.load.load_image as li


def reduce_images(project_folder: str, border: int) -> None:
    """
    This function removes a border from all images and masks in the project folder. This can
    sometimes be useful as otherwise parts of the border will appear in the 3D model. Note that only
    the resampled images will be affected by this change, the original images will remain unchanged.
    Args:
        project_folder (str): Path to the project folder
        border (int): Size of the border in px to remove from each side of the image
    Returns:
        None
    """

    # get all tif files in project folder that start with "OIS-"
    images = glob.glob(f"{project_folder}/OIS-*.tif")

    # add all images that are in the folder mask
    images += glob.glob(f"{project_folder}/masks/*.tif")

    for image_path in images:
        # Load the image
        img = li.load_image(image_path)

        # Get the size of the image
        height, width = img.shape

        # Define the size of the border to remove
        border = int(border)

        # Remove the border from the image
        img = img[border:height - border, border:width - border]

        # Save the image
        et.export_tiff(img, image_path, overwrite=True)
