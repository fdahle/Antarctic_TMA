
import src.load.load_image as li
import src.display.display_images as di

image_id="CA216132V0050"

image = li.load_image(image_id)

path_segmented = f"/data/ATM/data_1/aerial/TMA/segmented/unet/{image_id}.tif"
segmented = li.load_image(path_segmented)
segmented = segmented-1
di.display_images([image, segmented], image_types=["gray", "segmented"])