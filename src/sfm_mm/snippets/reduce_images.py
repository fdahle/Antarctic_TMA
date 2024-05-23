import glob

import src.export.export_tiff as et
import src.load.load_image as li


def reduce_images(project_folder, border):

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