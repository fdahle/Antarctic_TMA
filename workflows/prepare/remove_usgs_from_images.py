import os

import src.prepare.remove_usgs as ru
import src.load.load_image as li
import src.export.export_tiff as et

# Constants
PATH_IMAGE_FLD = "/data/ATM/data_1/aerial/TMA/downloaded/"

image_ids = ["CA214732V0030", "CA214732V0031", "CA214732V0032"]

def remove_usgs_from_images(image_ids):

    for image in image_ids:

        # define path to the image
        image_pth = os.path.join(PATH_IMAGE_FLD, image + ".tif")

        # load the image
        image = li.load_image(image_pth)

        # remove the logo
        image = ru.remove_usgs(image)

        if image is not None:

            # export the image
            et.export_tiff(image, image_pth, overwrite=True)

if __name__ == "__main__":
    remove_usgs_from_images(image_ids)