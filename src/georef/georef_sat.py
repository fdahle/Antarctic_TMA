import copy
import cv2
import numpy as np
import scipy

from typing import Optional, Tuple, Union
from shapely.geometry import Polygon
from shapely.wkt import loads as load_wkt

# import base functions
import src.base.enhance_image as ei
import src.base.find_tie_points as ftp
import src.base.rotate_image as ri
import src.base.rotate_points as rp

# import display functions
import src.display.display_images as di

# import georef snippet functions
import src.georef.snippets.calc_transform as ct

# import loading functions
import src.load.load_satellite as ls

debug_display_steps = False


class GeorefSatellite:

    def __init__(self,
                 min_tps_final: int = 25,
                 locate_image: bool = True, location_max_order: int = 3, location_overlap: float = 1 / 3,
                 tweak_image: bool = True, tweak_max_iterations: int = 10,
                 tweak_step_size: int = 2500, tweak_max_counter: int = 2,
                 enhance_image: bool = True, transform_method: str = "rasterio", transform_order: int = 3,
                 filter_outliers: bool = True):
        """
        Initialize the GeorefSatellite class with various settings for geo-referencing historical images
        with modern geo-referenced satellite images.
        Args:
            min_tps_final (int): Minimum number of tie points required for the final geo-referencing process.
                Defaults to 25.
            locate_image (bool): Whether to attempt locating the image within a broader area. Defaults to True.
            location_max_order (int): The maximum order to attempt for image location, affecting the search breadth.
                Defaults to 3.
            location_overlap (float): The overlap fraction between search tiles, influencing the granularity of the
                search. Defaults to 1/3.
            tweak_image (bool): Whether to refine the image positioning for better tie point matching. Defaults to True.
            tweak_max_iterations (int): The maximum number of iterations for tweaking the image position.
                Defaults to 10.
            tweak_step_size (int): The step size in pixels for each tweak iteration. Defaults to 2500.
            tweak_max_counter (int): The maximum count of non-improving tweaks before stopping. Defaults to 2.
            enhance_image (bool): Whether to enhance the image before tie point detection, which can improve matching.
                Defaults to True.
            transform_method (str): The method used for the transformation calculation. Defaults to "rasterio".
            transform_order (int): The order of transformation for the geo-referencing process. Defaults to 3.
            filter_outliers (bool): Whether to filter outliers in the tie point matching process. Defaults to True.
        """

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
        self.tweak_max_counter = tweak_max_counter

        # settings for filtering
        self.filter_outliers = filter_outliers

        # settings for enhance image
        self.enhance_image = enhance_image

        # transform settings
        self.transform_method = transform_method
        self.transform_order = transform_order

        # initialize tie point detector
        self.tp_finder = ftp.TiePointDetector(matching_method="lightglue")

    def georeference(self, input_image: np.ndarray, approx_footprint: Union[Polygon, str],
                     mask: Optional[np.ndarray] = None, angle: float = 0,
                     month: int = 0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                              Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Geo-references an input image by aligning it with a satellite image based on the provided approximate footprint,
        optional mask, rotation angle, and month for satellite imagery. This method uses tie-point matching to find
        corresponding points between the input image and the satellite image, and then calculates a transformation.
        Args:
            input_image (np.ndarray): The image to be geo-referenced as a NumPy array.
            approx_footprint (Union[Polygon, str]): An object representing the approximate footprint of the input image,
                as either a Shapely Polygon or a WKT string.
            mask (Optional[np.ndarray], optional): An optional mask image to be applied to the input image, as a NumPy
                array. Defaults to None.
            angle (float, optional): The rotation angle in degrees to be applied to the image and mask. Defaults to 0.
            month (int, optional): The month for which the satellite image is to be loaded. This parameter is used to
                select the appropriate satellite imagery. Defaults to 0.
        Returns:
            transform (Affine): The transformation matrix of the geo-referenced image as a NumPy array.
            residuals(np.ndarray): The residuals of the transformation points as a NumPy array.
            tps (np.ndarray): The tie points between the input image and the satellite image as a NumPy array.
            conf (np.ndarray): The confidence scores associated with the tie points as a NumPy array.
        Raises:
            Exception: If no satellite image is available for the given bounds and month.
        """

        print("Geo-reference image by satellite")

        image = copy.deepcopy(input_image)

        # check the input parameters
        if mask is not None:
            # TODO  implement
            pass

        # Check if approx_footprint is a WKT string, then convert it to a Shapely Polygon
        if isinstance(approx_footprint, str):
            approx_footprint = load_wkt(approx_footprint)

        # Ensure approx_footprint is now a Shapely Polygon
        if not isinstance(approx_footprint, Polygon):
            raise ValueError("approx_footprint must be a Shapely Polygon or a WKT string")

        # get the bounds of the approximate footprint
        image_bounds = approx_footprint.bounds

        # load the initial satellite image and transform
        sat, sat_transform = ls.load_satellite(image_bounds, month=month,
                                               return_empty_sat=True)

        # get the initial sat bounds
        sat_bounds = list(approx_footprint.bounds)
        sat_bounds = [int(x) for x in sat_bounds]

        # if we don't have a satellite image, we cannot geo-reference
        if sat is None:
            print("No satellite is available for this image")
            return None, None, None, None

        # rotate image (and mask)
        image, rotation_matrix = ri.rotate_image(image, angle)
        if mask is not None:
            mask, _ = ri.rotate_image(mask, angle)

        # adjust the image and mask so that it has the same pixel size as the satellite image
        image, adjust_factors = self._adjust_image_resolution(sat, image, sat_bounds, image_bounds)
        if mask is not None:
            mask, _ = self._adjust_image_resolution(sat, mask, sat_bounds, image_bounds)

        # enhance the image for an improved tie-point detection
        # TODO: FIX ENHANCE IMAGE; IF TRUE LESS TIE-points are found
        if self.enhance_image:
            image = ei.enhance_image(image, mask)

        # initial tie-point matching
        tps, conf = self.tp_finder.find_tie_points(sat, image, None, mask)

        if debug_display_steps:
            style_config = {"title": "Initial tie-points for geo-referencing"}
            di.display_images([sat, image], tie_points=tps, tie_points_conf=conf, style_config=style_config)

        # locate the image around the approx footprint
        if self.locate_image:
            sat, sat_bounds, sat_transform, tps, conf = self._perform_locate_image(image, mask, sat, sat_bounds,
                                                                                   sat_transform, tps, conf)

            if debug_display_steps:
                style_config = {"title": "Located tie-points for geo-referencing"}
                di.display_images([sat, image], tie_points=tps, tie_points_conf=list(conf), style_config=style_config)

        # tweak the image coordinates for maximum tie-points (minimum of 2 points required)
        if self.tweak_image and tps.shape[0] > 1:
            sat, sat_bounds, sat_transform, tps, conf = self._perform_tweak_image(image, mask, sat, sat_bounds,
                                                                                  sat_transform, tps, conf)

            if debug_display_steps:
                style_config = {"title": "Tweaked tie-points for geo-referencing"}
                di.display_images([sat, image], tie_points=tps, tie_points_conf=list(conf), style_config=style_config)

        # final check if there are enough tie-points
        if tps.shape[0] < self.min_tps_final:
            print(f"Too few tie-points found ({tps.shape[0]}/{self.min_tps_final}) for geo-referencing")
            return None, None, tps, conf

        # last filtering of the tie-points
        if self.filter_outliers:
            _, filtered = cv2.findHomography(tps[:, 0:2], tps[:, 2:4], cv2.RANSAC, 5.0)
            filtered = filtered.flatten()

            # 1 means outlier
            tps = tps[filtered == 0]
            conf = conf[filtered == 0]

            print(f"{np.count_nonzero(filtered)} outliers removed with RANSAC")

        # adjust points for the adapted image resolution
        tps[:, 2] = tps[:, 2] * (1 / adjust_factors[0])
        tps[:, 3] = tps[:, 3] * (1 / adjust_factors[1])

        # rotate points back for original image
        tps[:, 2:] = rp.rotate_points(tps[:, 2:], rotation_matrix, invert=True)

        # convert tie-points to absolute values
        absolute_points = np.array([sat_transform * tuple(point) for point in tps[:, 0:2]])
        tps[:, 0:2] = absolute_points

        # calculate the transformation-matrix
        transform, residuals = ct.calc_transform(input_image, tps,
                                                 transform_method=self.transform_method,
                                                 gdal_order=self.transform_order)

        return transform, residuals, tps, conf

    @staticmethod
    def _adjust_image_resolution(img1: np.ndarray, img2: np.ndarray, img_bound1: list[int],
                                 img_bound2: list[int]) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Adjusts the resolution of the second image to match that of the first image based on their respective bounds.
        Args:
            img1 (np.ndarray): The reference image with known resolution.
            img2 (np.ndarray): The image to be adjusted.
            img_bound1 (List[int]): The bounds of the first image (x_min, y_min, x_max, y_max).
            img_bound2 (List[int]): The bounds of the second image (x_min, y_min, x_max, y_max).
        Returns:
            resampled_img2 (np.ndarray): The adjusted image
            zoom_factors (Tuple[float, float]): The zoom factors (zoom_factor_x, zoom_factor_y) used for adjustment.
        """

        # deepcopy image 2 to not change the original
        img2 = copy.deepcopy(img2)

        # Extract image dimensions
        img_height_1, img_width_1 = img1.shape[1:]
        img_height_2, img_width_2 = img2.shape

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
        resampled_img2 = scipy.ndimage.zoom(img2, (zoom_factor_y, zoom_factor_x))

        print(f"Adjusted image resolution with "
              f"zoom-factor ({round(zoom_factor_y, 4) }, {round(zoom_factor_x, 4)})")

        return resampled_img2, (zoom_factor_x, zoom_factor_y)

    @staticmethod
    def _find_footprint_direction(img: np.ndarray, points: np.ndarray, step_size: int) -> Tuple[int, int]:
        """
        Finds the direction to adjust the footprint based on the distribution of tie points.
        Args:
            img (np.ndarray): The image for which the footprint direction is being determined.
            points (np.ndarray): An array of tie points.
            step_size (int): The step size in pixels to adjust the footprint.
        Returns:
            step_y (int): The adjustment in y direction.
            step_x (int): The adjustment in x direction.
        """

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

        # no step change if no points available
        if points.shape[0] == 0:
            return 0, 0

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

    def _perform_locate_image(self, img: np.ndarray, mask: Optional[np.ndarray], sat: np.ndarray,
                              sat_bounds: list[int], sat_transform: np.ndarray, tps: np.ndarray,
                              conf: np.ndarray) -> Tuple[np.ndarray, list[int], np.ndarray, np.ndarray, np.ndarray]:
        """
        Attempts to locate the image within the satellite imagery by adjusting its position to maximize
        the number of tie points. This method iterates through different positions around the initial guess,
        trying to find a location with a higher number of relevant tie points, which indicate a better alignment
        between the input image and the satellite imagery.
        Args:
            img (np.ndarray): The input image to be geo-referenced.
            mask (Optional[np.ndarray]): An optional mask image to be applied, ignoring tie-points from the masked
                pixels. Defaults to None.
            sat (np.ndarray): The initial satellite image against which the input image will be compared.
            sat_bounds (List[int]): The bounds of the initial satellite image (x_min, y_min, x_max, y_max).
            sat_transform (np.ndarray): The transformation matrix associated with the initial satellite image.
            tps (np.ndarray): The initial tie points found between the input image and the satellite image.
            conf (np.ndarray): The confidence scores associated with the initial tie points.
        Returns:
            sat (np.ndarray): The updated satellite image that best matches the input image after location adjustment.
            sat_bounds (List[int]): The updated bounds of the satellite image after the location adjustment.
            sat_transform (np.ndarray): The updated transformation matrix of the satellite image after location
                adjustment.
            tps (np.ndarray): The updated tie points between the input image and the best-matching satellite image.
            conf (np.ndarray): The updated confidence scores associated with the updated tie points.
        """

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

                print(f"  Check tile {len(lst_checked_tiles) - 1} (Coords: {tile}, Order {order})")

                # adapt the satellite bounds for this tile
                sat_bounds_tile = copy.deepcopy(sat_bounds)

                # first adapt x
                sat_bounds_tile[0] = sat_bounds_tile[0] + tile[0]
                sat_bounds_tile[2] = sat_bounds_tile[2] + tile[0]

                # and then y
                sat_bounds_tile[1] = sat_bounds_tile[1] + tile[1]
                sat_bounds_tile[3] = sat_bounds_tile[3] + tile[1]

                # get the satellite image
                sat_tile, sat_transform_tile = ls.load_satellite(sat_bounds_tile,
                                                                 return_empty_sat=True)

                if sat_tile is None:
                    print("  No satellite image could be found for this tile")
                    continue

                # get tie-points between satellite image and historical image
                tps_tile, conf_tile = self.tp_finder.find_tie_points(sat_tile, img,
                                                                     None, mask)

                print(f"  {tps_tile.shape[0]} points were found in this tile.")

                # if we didn't find tie-points we can immediately continue to the next tile
                if tps_tile.shape[0] == 0:
                    continue

                # check if we have the best combination
                if tps_tile.shape[0] > best_tps.shape[0] or \
                        (tps_tile.shape[0] == best_tps.shape[0] and
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

                # early stopping of the order search if the number of tie-points is very high
                if best_tps.shape[0] > self.min_tps_final * 25:
                    break

            # early stopping of general search if the number of tie-points is very high
            if best_tps.shape[0] > self.min_tps_final * 25:
                break

        # extract best values
        tps = copy.deepcopy(best_tps)
        conf = copy.deepcopy(best_conf)
        sat = copy.deepcopy(best_sat)
        sat_transform = copy.deepcopy(best_sat_transform)
        sat_bounds = copy.deepcopy(best_sat_bounds)

        # manually set conf to 0 if no tie-points were found
        if tps.shape[0] == 0:
            conf = np.zeros([0])
            conf_mean = 0
        else:
            # get average conf
            # noinspection PyTypeChecker
            conf_mean: float = np.mean(conf)

        print(f"Best tile is {best_tile} with {tps.shape[0]} tie-points ({round(conf_mean, 3)})")

        return sat, sat_bounds, sat_transform, tps, conf

    def _perform_tweak_image(self, img: np.ndarray, mask: Optional[np.ndarray], sat: np.ndarray,
                             sat_bounds: list[int], sat_transform: np.ndarray, tps: np.ndarray,
                             conf: np.ndarray) -> Tuple[np.ndarray, list[int], np.ndarray, np.ndarray, np.ndarray]:
        """
        This method tweaks the position of the input image by adjusting its coordinates in small increments,
        aiming to find a better alignment with the satellite image based on the distribution of tie points.
        Args:
            img (np.ndarray): The input image to be geo-referenced.
            mask (Optional[np.ndarray]): An optional mask image to be applied, ignoring tie-points from the masked
                pixels. Defaults to None.
            sat (np.ndarray): The satellite image against which the input image will be compared.
            sat_bounds (List[int]): The bounds of the satellite image (x_min, y_min, x_max, y_max).
            sat_transform (np.ndarray): The transformation matrix associated with the satellite image.
            tps (np.ndarray): The initial tie points found between the input image and the satellite image.
            conf (np.ndarray): The confidence scores associated with the initial tie points.
        Returns:
            sat (np.ndarray): The updated satellite image that best matches the input image after location adjustment.
            sat_bounds (List[int]): The updated bounds of the satellite image after the location adjustment.
            sat_transform (np.ndarray): The updated transformation matrix of the satellite image after location
                adjustment.
            tps (np.ndarray): The updated tie points between the input image and the best-matching satellite image.
            conf (np.ndarray): The updated confidence scores associated with the updated tie points.
        """

        print("Tweak image position")

        # get the start values for tweaking
        tweaked_sat = copy.deepcopy(sat)
        tweaked_sat_bounds = copy.deepcopy(sat_bounds)
        # tweaked_sat_transform = copy.deepcopy(sat_transform) will be overwritten in the first tweak
        tweaked_tps = copy.deepcopy(tps)
        # tweaked_conf = copy.deepcopy(conf)  # will be overwritten in the first tweak

        # get best values
        best_sat = copy.deepcopy(sat)
        best_sat_bounds = copy.deepcopy(sat_bounds)
        best_sat_transform = copy.deepcopy(sat_transform)
        best_tps = copy.deepcopy(tps)
        best_conf = copy.deepcopy(conf)

        # init counters for number of tweaks
        counter = 0
        counter_going_down = 0

        # tweak the image in a loop
        while counter < self.tweak_max_iterations:

            # increase tweak counter
            counter = counter + 1

            # find the direction in which we need to change the satellite image
            step_y, step_x = self._find_footprint_direction(tweaked_sat, tweaked_tps[:, 0:2],
                                                            self.tweak_step_size)

            print(f"  Tweak image ({counter}/{self.tweak_max_iterations}) with ({step_y}, {step_x})")

            # tweak the satellite bounds
            tweaked_sat_bounds[0] = tweaked_sat_bounds[0] + step_x
            tweaked_sat_bounds[1] = tweaked_sat_bounds[1] + step_y
            tweaked_sat_bounds[2] = tweaked_sat_bounds[2] + step_x
            tweaked_sat_bounds[3] = tweaked_sat_bounds[3] + step_y

            # get the tweaked satellite image
            tweaked_sat, tweaked_sat_transform = ls.load_satellite(tweaked_sat_bounds,
                                                                   return_empty_sat=True)

            if tweaked_sat is None:
                print("  Tweaked satellite image is not available")
                break

            # get tie-points between satellite image and historical image
            tweaked_tps, tweaked_conf = self.tp_finder.find_tie_points(tweaked_sat, img,
                                                                       None, mask)

            print(f"  {tweaked_tps.shape[0]} points found in tweak {counter} of "
                  f"{self.tweak_max_iterations}")

            # if the number of points is going up we need to save the params
            if tweaked_tps.shape[0] > best_tps.shape[0]:
                # save the best params
                best_sat = copy.deepcopy(tweaked_sat)
                best_sat_bounds = copy.deepcopy(tweaked_sat_bounds)
                best_sat_transform = copy.deepcopy(tweaked_sat_transform)
                best_tps = copy.deepcopy(tweaked_tps)
                best_conf = copy.deepcopy(tweaked_conf)

                # reset going down counter
                counter_going_down = 0

            else:

                print(f"  Points not increasing ({tweaked_tps.shape[0]} < {best_tps.shape[0]})")
                counter_going_down += 1

                if counter_going_down == self.tweak_max_counter:
                    # stop the loop
                    print("  Break the loop")

                    break

        # restore the best settings again
        sat = copy.deepcopy(best_sat)
        sat_bounds = copy.deepcopy(best_sat_bounds)
        sat_transform = copy.deepcopy(best_sat_transform)
        tps = copy.deepcopy(best_tps)
        conf = copy.deepcopy(best_conf)

        print(f"Tweaking finished with {tps.shape[0]} tie-points")

        return sat, sat_bounds, sat_transform, tps, conf
