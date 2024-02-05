import copy
import numpy as np
import scipy

import load.load_satellite as ls

import base.enhance_image as ei
import base.find_tie_points as ftp
import base.rotate_image as ri
import base.rotate_points as rp

import georef.snippets.calc_transform as ct
import georef.snippets.convert_image_to_footprint as citf


class GeorefSatellite:

    def __init__(self,
                 min_tps_final=25,
                 locate_image=True, location_max_order=3, location_overlap=1 / 3,
                 tweak_image=True, tweak_max_iterations=10, tweak_step_size=2500,
                 enhance_image=True,
                 transform_method="rasterio", transform_order=3):

        # settings for tps
        self.min_tps_final = min_tps_final

        # settings for locate_image
        self.locate_image = locate_image
        self.location_max_order = location_max_order
        self.location_overlap = location_overlap

        # settings for tweak image
        self.tweak_image = tweak_image
        self.tweak_max_iterations = tweak_max_iterations
        self.tweak_step_size = tweak_step_size

        # settings for enhance image
        self.enhance_image = enhance_image

        # initialize tie point detector
        self.tp_finder = ftp.TiePointDetector(matching_method="lightglue")

        self.transform_method = transform_method
        self.transform_order = transform_order

    def georeference(self, input_image, approx_footprint, mask=None, angle=0, month=0):

        print("Geo-reference image by satellite")

        image = copy.deepcopy(input_image)

        # check the input parameters
        if mask is not None:
            pass

        # get the bounds of the approximate footprint
        image_bounds = approx_footprint.bounds()

        # load the initial satellite image and transform
        sat, sat_transform = ls.load_satellite(image_bounds, month=month)

        # get the initial sat bounds
        sat_bounds = list(approx_footprint.bounds)
        sat_bounds = [int(x) for x in sat_bounds]

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
        image, adjust_factors = self._adjust_image_resolution(image, sat, image_bounds, sat_bounds)
        mask, _ = self._adjust_image_resolution(mask, sat, image_bounds, sat_bounds)

        # enhance the image for an improved tie-point detection
        if self.enhance_image:
            image = ei.enhance_image(image, mask)

        # initial tie-point matching
        tps, conf = self.tp_finder.find_tie_points(sat, image, None, mask)

        # locate the image around the approx footprint
        if self.locate_image:
            sat, sat_transform, sat_bounds, tps, conf = self._perform_locate_image(image, mask, sat, sat_bounds,
                                                                                   sat_transform, tps, conf)
        # tweak the image coordinates for maximum tie-points
        if self.tweak_image:
            sat, sat_transform, sat_bounds, tps, conf = self._perform_tweak_image(image, mask, sat, sat_bounds,
                                                                                  sat_transform, tps, conf)

        # final check if there are enough tie-points
        if tps.shape[0] < self.min_tps_final:
            print(f"Too few tie-points found ({tps.shape[0]}/{self.min_tps_final}) for geo-referencing")

        # adjust points for the adapted image resolution
        tps[:, 2] = tps[:, 2] * (1 / adjust_factors[0])
        tps[:, 3] = tps[:, 3] * (1 / adjust_factors[1])

        # rotate points back for original image
        tps = rp.rotate_points(tps, -angle, new_center="", original_center="")

        # convert tie-points to absolute values
        absolute_points = np.array([sat_transform * tuple(point) for point in tps[:, 0:2]])
        tps[:, 0:2] = absolute_points

        transform, residuals = ct.calc_transform(input_image, tps,
                                                 transform_method=self.transform_method,
                                                 gdal_order=self.transform_order)

        footprint = citf.convert_image_to_footprint(image, transform=transform)

        return footprint, transform, residuals

    @staticmethod
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

    @staticmethod
    def _find_footprint_direction(img, points, step_size):

        # Validate input image dimensions
        if img.ndim not in [2, 3]:
            raise ValueError(f"Unsupported image dimensions: {img.shape}")
        if img.ndim == 3 and 3 not in img.shape[:2] and 3 != img.shape[2]:
            raise ValueError(f"Unsupported image channel format: {img.shape}")

        # Determine image dimensions based on input shape
        if img.ndim == 3 and img.shape[2] == 3:
            image_height, image_width = img.shape[:2]
        else:
            image_height, image_width = img.shape[-2:]

        # Calculate midpoints for the image
        mid_x, mid_y = image_width // 2, image_height // 2

        # Determine indices of points in each quadrant
        idx_q1 = np.where((points[:, 0] <= mid_x) & (points[:, 1] <= mid_y))[0]
        idx_q2 = np.where((points[:, 0] > mid_x) & (points[:, 1] <= mid_y))[0]
        idx_q3 = np.where((points[:, 0] <= mid_x) & (points[:, 1] > mid_y))[0]
        idx_q4 = np.where((points[:, 0] > mid_x) & (points[:, 1] > mid_y))[0]

        # Calculate percentage of points in each quadrant
        total_points = len(points)
        perc_q1, perc_q2, perc_q3, perc_q4 = (len(q) / total_points for q in [idx_q1, idx_q2, idx_q3, idx_q4])

        # Determine step adjustments based on point distribution
        step_x = int((np.average([perc_q2, perc_q4]) - np.average([perc_q1, perc_q3])) * step_size)
        step_y = int((np.average([perc_q1, perc_q2]) - np.average([perc_q3, perc_q4])) * step_size)

        return step_y, step_x

    def _perform_locate_image(self, img, mask, sat, sat_bounds, sat_transform, tps, conf):

        print("Locate image position")

        # get the width and height of the satellite image
        sat_width = sat.shape[1] * np.abs(sat_transform[0])
        sat_height = sat.shape[2] * np.abs(sat_transform[4])

        # calculate the step size
        step_x = int((1 - self.location_overlap) * sat_width)
        step_y = int((1 - self.location_overlap) * sat_height)

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
        for order in range(1, self.location_max_order):

            # create a tile dict
            tiles = []
            for i in range(-order * step_x, (order + 1) * step_x, step_x):
                for j in range(-order * step_y, (order + 1) * step_y, step_y):
                    tiles.append([i, j])

            # check all tiles
            for tile in tiles:

                # check if we already checked this tile
                if tile in lst_checked_tiles:
                    continue

                # add the tile to the list of checked tiles
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
                sat_tile, sat_transform_tile = ls.load_satellite(sat_bounds_tile)

                if sat_tile is None:
                    print("  No satellite image could be found for this tile")
                    continue

                # get tie-points between satellite image and historical image
                tps_tile, conf_tile = self.tp_finder.find_tie_points(sat_tile, img,
                                                                     None, mask)

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

                # early stopping if the number of tie-points is very high
                if best_tps.shape[0] > self.min_tps_final * 25:
                    break

        # extract best values
        tps = copy.deepcopy(best_tps)
        conf = copy.deepcopy(best_conf)
        sat = copy.deepcopy(best_sat)
        sat_transform = copy.deepcopy(best_sat_transform)
        sat_bounds = copy.deepcopy(best_sat_bounds)

        print(f"Best tile is {best_tile} with {tps.shape[0]} tie-points ({np.mean(conf)})")

        return sat, sat_transform, sat_bounds, tps, conf

    def _perform_tweak_image(self, img, mask, sat, sat_bounds, sat_transform, tps, conf):

        print("Tweak image position")

        # get the start values for tweaking
        tweaked_sat = copy.deepcopy(sat)
        tweaked_sat_bounds = copy.deepcopy(sat_bounds)
        tweaked_sat_transform = copy.deepcopy(sat_transform)
        tweaked_tps = copy.deepcopy(tps)
        # tweaked_conf = copy.deepcopy(conf)  # will be overwritten in the first tweak

        # get best values
        best_sat = copy.deepcopy(sat)
        best_sat_transform = copy.deepcopy(sat_transform)
        best_sat_bounds = copy.deepcopy(sat_bounds)
        best_tps = copy.deepcopy(tps)
        best_conf = copy.deepcopy(conf)

        # init counter for number of tweaks
        counter = 0

        # tweak the image in a loop
        while counter < self.tweak_max_iterations:

            # increase tweak counter
            counter = counter + 1

            # find the direction in which we need to change the satellite image
            step_y, step_x = self._find_footprint_direction(tweaked_sat, tweaked_tps[:, 0:2],
                                                            self.tweak_step_size)

            print(f"Tweak image ({counter}/{self.tweak_max_iterations}) with ({step_y}, {step_x})")

            # tweak the satellite bounds
            tweaked_sat_bounds[0] = tweaked_sat_bounds[0] + step_x
            tweaked_sat_bounds[1] = tweaked_sat_bounds[1] + step_y
            tweaked_sat_bounds[2] = tweaked_sat_bounds[2] + step_x
            tweaked_sat_bounds[3] = tweaked_sat_bounds[3] + step_y

            # get the tweaked satellite image
            tweaked_sat, sat_transform_tweaked = ls.load_satellite(tweaked_sat_bounds)

            if tweaked_sat is None:
                print("  Tweaked satellite image is not available")
                break

            # get tie-points between satellite image and historical image
            tweaked_tps, tweaked_conf = self.tp_finder.find_tie_points(tweaked_sat, img,
                                                                       None, mask)

            print(f"  {tweaked_tps.shape[0]} points found in tweak {counter} of "
                  f"{self.tweak_max_iterations}")

            # if the number of points is going up we need to save the params
            if tweaked_tps.shape[0] >= best_tps.shape[0]:
                # save the best params
                best_sat = copy.deepcopy(tweaked_sat)
                best_sat_bounds = copy.deepcopy(tweaked_sat_bounds)
                best_sat_transform = copy.deepcopy(tweaked_sat_transform)
                best_tps = copy.deepcopy(tweaked_tps)
                best_conf = copy.deepcopy(tweaked_conf)

            else:
                # stop the loop
                print(f"  Points going down ({tweaked_tps.shape[0]} < {best_tps.shape[0]})")
                break

        # restore the best settings again
        sat = copy.deepcopy(best_sat)
        sat_bounds = copy.deepcopy(best_sat_bounds)
        sat_transform = copy.deepcopy(best_sat_transform)
        tps = copy.deepcopy(best_tps)
        conf = copy.deepcopy(best_conf)

        print(f"Tweaking finished with {tps.shape[0]} tie-points")

        return sat, sat_transform, sat_bounds, tps, conf
