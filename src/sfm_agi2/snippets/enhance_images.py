import os
from tqdm import tqdm

import src.base.enhance_image as ei
import src.export.export_tiff as eti
import src.load.load_image as li

def enhance_images(enhanced_folder,
                   image_folder, mask_folder,
                   image_names,
                   use_cached_images: bool = False,
                   save_cached_images: bool = False,
                   cache_folder: str | None = None) -> None:

    # init variable
    enhanced_image = None

    # create folder for enhanced images
    os.makedirs(enhanced_folder, exist_ok=True)

    # create progressbar
    pbar = tqdm(total=len(image_names), desc="Enhance images",
                position=0, leave=True)

    # iterate all tif files in image folder
    for image_id in image_names:
        # update the progress bar description
        pbar.set_postfix_str(image_id)

        # flag to check if the image is loaded from cache
        cached_image_loaded = False

        # check if there is a cached version
        if cache_folder is not None and use_cached_images:
            cache_path = os.path.join(cache_folder, image_id + ".tif")
            if os.path.exists(cache_path):
                enhanced_image = li.load_image(cache_path)
                cached_image_loaded = True

        if cached_image_loaded is False:
            # get the image path
            img_pth = os.path.join(image_folder, image_id + ".tif")

            # load the image
            image = li.load_image(img_pth)

            # load the mask
            mask_path = os.path.join(mask_folder, image_id + ".tif")
            mask = li.load_image(mask_path)

            # enhance the image
            enhanced_image = ei.enhance_image(image, mask)

            # save the enhanced image in cache
            if save_cached_images and cache_folder is not None:
                cache_path = os.path.join(cache_folder, image_id + ".tif")
                eti.export_tiff(enhanced_image, cache_path, use_lzw=True)

        # save the enhanced image
        enhanced_path = os.path.join(enhanced_folder, image_id + ".tif")
        eti.export_tiff(enhanced_image, enhanced_path, use_lzw=True)

        # update progress bar
        pbar.update(1)

    # close progress bar
    pbar.set_postfix_str("- Finished -")
    pbar.close()
