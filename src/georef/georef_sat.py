import copy
import numpy as np
import scipy

import load.load_satellite as ls

import base.enhance_image as ei
import base.find_tie_points as ftp
import base.rotate_image as ri
import base.rotate_points as rp

import georef.snippets.calc_transform as ct

def georef_sat(input_image, approx_footprint, mask=None, angle=0, month=0,
               locate_image=True, location_max_order=3, location_overlap=1/3,
               tweak_image=True, tweak_max_iterations = 10,
               enhance_image=True):

    print("Geo-reference image by satellite")

    image = copy.deepcopy(input_image)

    # check the input parameters
    if mask is not None:
        pass

    # get the bounds of the approximate footprint
    image_bounds = approx_footprint.bounds()

    # load the initial satellite image and it's bounds
    sat, sat_bounds = ls.load_satellite(image_bounds, month=month)

    # if we don't have a satellite image, we cannot geo-reference
    if sat is None:
        print("No satellite is available for this image")
        return

    # rotate image (and mask)
    # (IMAGE CHANGE NR. 1)
    image, _, _ = ri.rotate_image(image, angle)
    if mask is not None:
        mask, _, _ = ri.rotate_image(mask, angle)

    # adjust the image and mask so that it has the same pixel size as the satellite image
    # (IMAGE CHANGE NR. 2)
    image, adjust_factors = _adjust_image_resolution(image, sat, image_bounds, sat_bounds)
    mask, _ = _adjust_image_resolution(mask, sat, image_bounds, sat_bounds)

    # enhance the image for an improved tie-point detection
    if enhance_image:
        image = ei.enhance_image(image, mask)

    # initialize tie point detector
    tp_finder = ftp.TiePointDetector()

    # initial tie-point matching
    tps, conf = tp_finder.find_tie_points(sat, image, None, mask)

    # locate the image around the approx footprint
    _locate_image(image, sat, sat_bounds, sat_transform, tps, conf)

    # tweak the image coordinates for maximum tie-points
    _tweak_image()

    # final check if there are enough tie-points


    # we need to adapt the tie-points to account for changes in the image
    # first for adjust factor

    # then for rotation

    # convert tie-points to absolute values


    transform, residuals = ct.calc_transform(tps)

    return footprint, transform, residuals

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

def _locate_image(img, sat, sat_bounds, sat_transform,
                  tps, conf, max_order, overlap):

    print("Locate image position")

    # get the width and height of the satellite image
    sat_width = sat.shape[1] * np.abs(sat_transform[0])
    sat_height = sat.shape[2] * np.abs(sat_transform[4])

    # calculate the step size
    step_x = int((1 - overlap) * sat_width)
    step_y = int((1 - overlap) * sat_height)

    # list of already checked  tiles (center is always already checked in the init)
    lst_checked_tiles = [[0, 0]]

    # save the best values (start with initial values from center)
    best_tps = tps
    best_conf = conf
    best_tile = [0, 0]

    # best satellite
    best_sat = copy.deepcopy(sat)
    best_sat_transform = copy.deepcopy(sat_transform)
    best_sat_bounds = copy.deepcopy(sat_bounds)

    # iterate the images around our image (ignoring the center)
    for order in range(1, max_order):

        # create a combinations dict
        tiles = []
        for i in range(-order * step_x, (order + 1) * step_x, step_x):
            for j in range(-order * step_y, (order + 1) * step_y, step_y):
                tiles.append([i, j])

        # check all combinations
        for tile in tiles:

            # check if we already checked this tile
            if tile in lst_checked_tiles:
                continue

            # add the combination to the list of checked combinations
            lst_checked_tiles.append(tile)

            print(f"Check tile {len(lst_checked_tiles)} (Coords: {tile}, Order {order})")

            # adapt the satellite bounds for this tile
            sat_bounds_tile = copy.deepcopy(sat_bounds)

            # first adapt x
            sat_bounds_tile[0] = sat_bounds_tile[0] + tile[0]
            sat_bounds_tile[2] = sat_bounds_tile[2] + tile[0]

            # and then y
            sat_bounds_tile[1] = sat_bounds_tile[1] + tile[1]
            sat_bounds_tile[3] = sat_bounds_tile[3] + tile[1]

            # get the satellite image
            sat_tile, sat_transform_tile = ls.load_satellite()

            if sat_tile is None:
                print("  No satellite image could be found for this tile")
                continue

            # get tie-points between satellite image and historical image
            tps_tile, conf_tile = ftp.find_tie_points()

            # check if we have the best combination
            if tps_tile > best_tps or \
                    (tps_tile == best_tps and
                     np.mean(conf_tile) > np.mean(best_conf)):

                # save best tie-points and conf
                best_tps = copy.deepcopy(tps_tile)
                best_conf = copy.deepcopy(conf_tile)

                # save best tile
                best_tile = copy.deepcopy(tile)

                # save best satellite information
                best_sat = copy.deepcopy(sat_tile)
                best_sat_transform = copy.deepcopy(sat_transform_tile)
                best_sat_bounds = copy.deepcopy(sat_bounds_tile)

            # early stopping if the number of tie-points is large enough
            if best_tps.shape[0] > min_tps_final * 25:
                break

    # extract best values
    tps = copy.deepcopy(best_tps)
    conf = copy.deepcopy(best_conf)
    sat = copy.deepcopy(best_sat)
    sat_transform = copy.deepcopy(best_sat_transform)
    sat_bounds = copy.deepcopy(best_sat_bounds)

    print(f"Best tile is {best_tile} with {tps.shape[0]} tie-points ({np.mean(conf)})")

    return tps, conf, sat, sat_transform, sat_bounds

def _tweak_image(img, sat, sat_bounds, sat_transform, tps, conf, max_iterations):

    # save the best values (start with initial values from center)
    best_tps = tps
    best_conf = conf

    # get the start values for tweaking
    sat_tweaked = copy.deepcopy(sat)
    sat_bounds_tweaked = copy.deepcopy(sat_bounds)
    tps_tweaked = copy.deepcopy(tps)

    # best satellite
    best_sat = copy.deepcopy(sat)
    best_sat_transform = copy.deepcopy(sat_transform)
    best_sat_bounds = copy.deepcopy(sat_bounds)

    # init counter for number of tweaks
    counter = 0

    # tweak the image in a loop
    while counter < max_iterations:

        # increase tweak counter
        counter = counter + 1

        # find the direction in which we need to change the satellite image
        step_y, step_x = ffd.find_footprint_direction(sat_tweaked, tps_tweaked[:, 0:2])

        print(f"Tweak image ({counter}/{max_iterations}) with ({step_y}, {step_x})")

        # tweak the satellite bounds
        sat_bounds_tweaked[0] = sat_bounds_tweaked[0] + step_x
        sat_bounds_tweaked[1] = sat_bounds_tweaked[1] + step_y
        sat_bounds_tweaked[2] = sat_bounds_tweaked[2] + step_x
        sat_bounds_tweaked[3] = sat_bounds_tweaked[3] + step_y


def _filter_tps():
    pass
