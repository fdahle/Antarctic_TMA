import json
import os
import scipy

import base.print_v as p


def adjust_image_resolution(image_1, bounds_1, image_2, bounds_2,
                            buffer_image_1=None,
                            verbose=False, pbar=None):
    """
    adjust_image_resolution(image_1, bounds_1, image_2, bounds_2, buffer_image, verbose, pbar):
    This function looks at two images with known width and height in real life (determined by the bounding
    box). The size of the second image is then changed so that both images have the same resolution (=meter
    per pixel). Hint: The quality is better if the second image is larger than the first image
    Args:
        image_1 (np-array): Numpy-array containing the base image
        bounds_1 (list): List describing the bounds of image_1 [min_x, min_y, max_x, max_y]
        image_2 (np-array): Numpy-array containing the image that should be adjusted
        bounds_2 (list): List describing the bounds of image_2 [min_x, min_y, max_x, max_y]
        buffer_image_1 (Boolean, None): TODO: CHECK THIS
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        resampled_image2 (np-array): image_2 as a numpy array, but with adjusted resolution
    """

    p.print_v("Start: adjust_image_resolution", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder[:-4] + "/params.json") as j_file:
        json_data = json.load(j_file)

    # set default buffer value if not specified
    if buffer_image_1 is None:
        buffer_image_1 = json_data["footprint_buffer"]

    # get height and with of the images
    if len(image_1.shape) == 3:
        img_height_1 = image_1.shape[1]
        img_width_1 = image_1.shape[2]
    else:
        img_height_1 = image_1.shape[0]
        img_width_1 = image_1.shape[1]
    if len(image_2.shape) == 3:
        img_height_2 = image_2.shape[1]
        img_width_2 = image_2.shape[2]
    else:
        img_height_2 = image_2.shape[0]
        img_width_2 = image_2.shape[1]

    # get height and width of the approx_footprint
    fp_width_1 = bounds_1[2] - bounds_1[0]
    fp_height_1 = bounds_1[3] - bounds_1[1]
    fp_width_2 = bounds_2[2] - bounds_2[0]
    fp_height_2 = bounds_2[3] - bounds_2[1]

    # count for buffer
    fp_width_1 = fp_width_1 - buffer_image_1 * 2
    fp_height_1 = fp_height_1 - buffer_image_1 * 2

    # Determine the pixel sizes of both images in x and y directions
    pixel_size_x1 = fp_width_1 / img_width_1
    pixel_size_y1 = fp_height_1 / img_height_1
    pixel_size_x2 = fp_width_2 / img_width_2
    pixel_size_y2 = fp_height_2 / img_height_2

    # Determine the zoom factor required to match the pixel sizes in x and y directions
    zoom_factor_x = pixel_size_x1 / pixel_size_x2
    zoom_factor_y = pixel_size_y1 / pixel_size_y2

    if zoom_factor_x > 0 or zoom_factor_y > 0:
        zoom_factor_x = 1 / zoom_factor_x
        zoom_factor_y = 1 / zoom_factor_y

    # Resample the first image using the zoom factors in x and y directions
    resampled_image2 = scipy.ndimage.zoom(image_2, (zoom_factor_y, zoom_factor_x))

    p.print_v("Finished: adjust_image_resolution", verbose=verbose, pbar=pbar)

    return resampled_image2
