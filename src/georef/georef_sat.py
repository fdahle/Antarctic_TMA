import copy
import scipy

import load.load_satellite as ls

def georef_sat(image, angle, approx_footprint):

    print("Geo-reference image by satellite")

    # get the bounds of the approximate footprint
    image_bounds = approx_footprint.bounds()

    # load the initial satellite image and it's bounds
    sat, sat_bounds = ls.load_satellite()

    # if we don't have a satellite image, we cannot geo-reference
    if sat is None:
        print("No satellite is available for this image")
        return

    # adjust the image so that it has the same pixel size as the satellite image
    image_adjusted, adjust_factors = _adjust_image_resolution(image, sat, image_bounds, sat_bounds)

    # initial tie-point matching
    tps, conf = ftp.find_tie_points(sat, image_enhanced,
                                    min_threshold=0,
                                    filter_outliers=False,
                                    additional_matching=True, extra_matching=True,


def _adjust_image_resolution(img1, img2, img_bound1, img_bound2):

    # deepcopy image 1 to not change the original
    img1 = copy.deepcopy(img1)

    # Extract image dimensions
    img_height_1, img_width_1 = img1.shape[:2]
    img_height_2, img_width_2 = img2.shape[:2]

    # get height and width from the image bounds
    fp_width_1 = img_bound1[2] - img_bound1[0]
    fp_height_1 = img_bound1[3] - img_bound1[1]
    fp_width_2 = img_bound2[2] - img_bound2[0]
    fp_height_2 = img_bound2[3] - img_bound2[1]

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
    resampled_img1 = scipy.ndimage.zoom(img1, (zoom_factor_y, zoom_factor_x))

    print("Adjusted image resolution from {} to {}")

    return resampled_img1, (zoom_factor_x, zoom_factor_y)

def _locate_image():

    pass


def _tweak_image():
    pass

def _filter_tps():
    pass
