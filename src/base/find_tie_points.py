import copy
import cv2
import math
import numpy as np
import torch
import warnings

from scipy.spatial.distance import cdist
from skimage import transform as tf
from typing import List, Tuple, Optional

import src.base.custom_print as cp
import src.base.resize_image as ri

import src.display.display_images as di

from external.lightglue import LightGlue, SuperPoint
from external.lightglue.utils import rbd
from external.SuperGlue.matching import Matching

OOM_REDUCE_VALUE = 0.9


# TODO: FIX WARNING LEVELS OF PRINT

class TiePointDetector:

    def __init__(self, matching_method: str, matching_additional: bool = True, matching_extra: bool = True,
                 keep_resized_points: bool = False, min_resized_points: int = 10, num_transform_points: int = 25,
                 min_conf_value: float = 0.0, ransac_value: float = 5.0, average_threshold: float = 10.0,
                 display: bool = False, catch=True, verbose: bool = True):
        """
        Initializes the TiePointDetector with specified configuration for tie-point detection and matching.

        Args:
            matching_method (str): The method used for matching tie points.
                Supported values are 'lightglue' and 'superglue'.
            matching_additional (bool, optional): If True, perform additional matching after the initial matching.
                Defaults to True.
            matching_extra (bool, optional): If True, perform extra matching after additional matching for potentially
                better results. Defaults to True.
            keep_resized_points (bool, optional): If True, keep tie points from resized images
                in the final results. Defaults to False.
            min_resized_points (int, optional): The minimum number of tie points required from resized images for
                additional matching. Defaults to 10.
            num_transform_points (int, optional): The number of points to use when calculating transformations for extra
                matching .Defaults to 25.
            min_conf_value (float, optional): The minimum confidence value for considering a match. Defaults to 0.0.
            ransac_value (float, optional): The RANSAC re-projection threshold. Defaults to 5.0.
            average_threshold (float, optional): The threshold for averaging tie points. Defaults to 10.0.
            verbose (bool, optional): If True, print detailed messages during processing. Defaults to True.
        """

        self.average_threshold = average_threshold
        self.keep_resized_points = keep_resized_points
        self.matching_method = matching_method.lower()
        self.matching_additional = matching_additional
        self.matching_extra = matching_extra
        self.min_conf_value = min_conf_value
        self.min_resized_points = min_resized_points
        self.num_transform_points = num_transform_points
        self.ransac_value = ransac_value

        self.verbose = verbose
        self.catch = catch
        self.display = display
        self.logger = cp.CustomPrint(verbosity=0)

        self.max_height = 2000
        self.max_width = 2000

        # check some settings
        if self.matching_extra:
            if self.matching_additional is False:
                raise ValueError("Additional matching must be enabled for extra matching")

        # Initialize device based on CUDA availability
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

        # init matcher
        self.matcher, self.extractor = self._init_matcher()

        self.logger.print(f"TiePointDetector initialized using {matching_method} on {self.device}", color="OKBLUE")

    def find_tie_points(self, input_img1, input_img2, mask1=None, mask2=None):
        """
        Finds tie points between two input images, optionally using masks to limit the search area.

        Args:
            input_img1: The first input image as a NumPy array.
            input_img2: The second input image as a NumPy array.
            mask1: Optional mask for the first image to specify areas of interest.
                Must match the dimensions of `input_img1`. 0 values are filtered, 1 values are kept
            mask2: Optional mask for the second image to specify areas of interest.
                Must match the dimensions of `input_img2`. 0 values are filtered, 1 values are kept

        Returns:
            A tuple containing two elements:
                - A NumPy array of the final averaged tie points.
                - A NumPy array of the confidence scores associated with each tie point.

        Raises:
            AssertionError: If the dimensions of the masks do not match their corresponding images.
        """

        # TODO: add rotation

        if mask1 is not None:
            assert (input_img1.shape[0] == mask1.shape[0] and
                    input_img1.shape[1] == mask1.shape[1])
        if mask2 is not None:
            assert (input_img2.shape[0] == mask2.shape[0] and
                    input_img2.shape[1] == mask2.shape[1])

        try:
            # we handle the warnings ourselves -> ignore them
            with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # Ignore runtime warnings, e.g., mean of empty slice

                # prepare the images
                img1, img2 = self._prepare_images(input_img1, input_img2)

                # initial tie-point matching
                tps, conf = self._perform_initial_matching(img1, img2)

                # mask initial tie-points
                tps, conf = self._mask_tie_points(tps, conf, mask1, mask2)

                # display the initial tie-points
                if self.display:
                    style_config = {"title": f"{len(conf)} initial Tie-points"}
                    di.display_images([img1, img2], tie_points=tps, tie_points_conf=list(conf),
                                      style_config=style_config)

                if tps.shape[0] < self.min_resized_points:
                    # print(f"Not enough resized tie-points found ({len(conf)} of {self.min_resized_points})")
                    return np.empty((0, 4)), np.empty((0, 1))

                # optional additional matching
                if self.matching_additional:
                    # additional tie-point matching
                    tps_additional, conf_additional = self._perform_additional_matching(img1, img2, tps)

                    # mask additional tie-points
                    tps_additional, conf_additional = self._mask_tie_points(tps_additional, conf_additional,
                                                                            mask1, mask2)

                    # display the additional tie-points
                    if self.display:
                        style_config = {"title": f"{len(conf_additional)} additional Tie-points"}
                        di.display_images([img1, img2],
                                          tie_points=tps_additional, tie_points_conf=list(conf_additional),
                                          style_config=style_config)

                    # either we keep the resized points or just use the additional ones
                    if self.keep_resized_points:
                        tps = np.concatenate((tps, tps_additional))
                        conf = np.concatenate((conf, conf_additional))
                    else:
                        tps = tps_additional
                        conf = conf_additional

                # sometimes there are too few tie-points after additional matching -> stop process
                if tps.shape[0] < 3:
                    return tps, conf

                # optional extra matching
                if self.matching_extra:
                    # extra tie-point matching
                    tps_extra, conf_extra = self._perform_extra_matching(img1, img2, mask1, mask2, tps, conf)

                    # mask extra tie-points
                    tps_extra, conf_extra = self._mask_tie_points(tps_extra, conf_extra, mask1, mask2)

                    # display the additional tie-points
                    if self.display:
                        style_config = {"title": f"{len(conf_extra)} extra Tie-points"}
                        di.display_images([img1, img2],
                                          tie_points=tps_extra, tie_points_conf=list(conf_extra),
                                          style_config=style_config)

                    # add the extra tie-points
                    tps = np.concatenate((tps, tps_extra))
                    conf = np.concatenate((conf, conf_extra))

                    # remove duplicates in tps and conf (can be that the same tie-points are detected
                    # in additional and extra)
                    tps, unique_indices = np.unique(tps, return_index=True, axis=0)
                    tps = tps.astype(int)
                    conf = conf[unique_indices]

                # apply threshold filter
                tps, conf = self._filter_with_threshold(tps, conf)

                # apply outlier filter
                tps, conf = self._filter_outliers(tps, conf)

                # average the tie-points for logical consistency
                # (same tie-points pointing to different other tie-points are removed)
                tps, conf = self._average_tie_points(tps, conf, [0, 1])
                tps, conf = self._average_tie_points(tps, conf, [2, 3])

                # display the final tie-points
                if self.display:
                    style_config = {"title": f"{len(conf)} final Tie-points"}
                    di.display_images([img1, img2],
                                      tie_points=tps, tie_points_conf=list(conf),
                                      style_config=style_config)
        except (Exception,) as e:
            if self.catch:
                return np.empty((0, 4)), np.empty((0, 1))
            else:
                raise e

        return tps, conf

    def _average_tie_points(self, tps: np.ndarray, conf: np.ndarray, cols: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Averages tie points based on the provided confidence scores and specified columns for key generation. This
        function calculates weighted averages of tie points that share the same key (generated based on specified
        columns). If the maximum pairwise distance between points sharing the same key is below a threshold,
        these points are averaged, otherwise these points are discarded.

        Args:
            tps: An array of tie points, where each row is a tie point.
            conf: A array of confidence scores corresponding to each tie point.
            cols: A list of columns to be used for generating unique keys to group tie points.

        Returns:
            A tuple of two numpy arrays:
            - The first array contains the new averaged tie points.
            - The second array contains the new confidence scores associated with these averaged tie points.

        """

        # Dictionary to hold unique keys and their x2, y2 or x1, y1 values for averaging
        avg_dict = {}
        for row, conf in zip(tps, conf):
            key = tuple(row[cols])
            if key not in avg_dict:
                avg_dict[key] = {'values': [], 'confs': []}
            avg_dict[key]['values'].append(row[2 - cols[0]:4 - cols[0]])
            avg_dict[key]['confs'].append(conf)

        new_tie_points = []
        new_confs = []
        for key, data in avg_dict.items():
            # Calculate the maximum pairwise distance between points
            distances = cdist(data['values'], data['values'], 'euclidean')
            max_distance = np.max(distances)

            # Check if the maximum distance exceeds the threshold
            if max_distance <= self.average_threshold:
                weighted_avg_values = np.average(data['values'], axis=0, weights=data['confs'])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Mean of empty slice.")
                    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
                    avg_conf = np.mean(data['confs'])
                if cols == [0, 1]:
                    new_row = list(key) + weighted_avg_values.tolist()
                else:
                    new_row = weighted_avg_values.tolist() + list(key)
                new_tie_points.append(new_row)
                new_confs.append(avg_conf)

        return np.array(new_tie_points), np.array(new_confs)

    def _filter_outliers(self, tps: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Filter outliers using RANSAC. This method uses RANSAC to filter outliers from the input arrays 'tps' and 'conf'.
        It identifies outliers using the 'cv2.findHomography' function and removes them from 'tps' and 'conf'.

        Args:
            tps (np.ndarray): Input array of data points.
            conf (np.ndarray): Input array of confidence values.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing filtered 'tps' and 'conf' arrays.
        """

        _, filtered = cv2.findHomography(tps[:, 0:2], tps[:, 2:4], cv2.RANSAC, self.ransac_value)
        filtered = filtered.flatten()

        # 1 means outlier
        tps = tps[filtered == 0]
        conf = conf[filtered == 0]

        self.logger.print(f"{np.count_nonzero(filtered)} outliers removed with RANSAC", color="OKBLUE")

        return tps, conf

    def _filter_with_threshold(self, tps: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Filter data based on a confidence threshold. This method filters the input arrays 'tps' and 'conf' based on the
        minimum confidence value. It removes elements in 'tps' and 'conf' where the corresponding confidence values
        are less than 'min_conf_value'.

        Args:
            tps (np.ndarray): Input array of data points.
            conf (np.ndarray): Input array of confidence values.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing filtered 'tps' and 'conf' arrays.
        """

        # filter tps and conf with minimum threshold
        tps = tps[conf >= self.min_conf_value]
        conf = conf[conf >= self.min_conf_value]

        self.logger.print(f"{np.count_nonzero(conf < self.min_conf_value)} outliers removed with "
                          f"Threshold {self.min_conf_value}", color="OKBLUE")

        return tps, conf

    def _init_matcher(self) -> Tuple[Optional[object], Optional[object]]:
        """
        Initializes and returns the matcher and feature extractor based on the specified matching method.
        This method supports initialization for two types of matching methods: "lightglue" and "superglue".
        For "lightglue", it initializes both a feature extractor and a matcher. For "superglue", it initializes
        only a matcher as feature extraction is integrated within the matching process.

        Returns:
            Tuple[Optional[object], Optional[object]]: A tuple containing the matcher and feature extractor.
                - The first element is the matcher object, which is initialized based on the matching method.
                - The second element is the feature extractor object, which is initialized only for "lightglue".
                  For "superglue", this element is None.

        Raises:
            ValueError: If an unsupported matching method is provided.

        Note:
            The method requires the `matching_method` and `device` attributes to be set beforehand.
            The returned objects are PyTorch models moved to the specified device.
        """

        if self.matching_method == "lightglue":
            # Initialize the feature extractor and matcher for LightGlue
            extractor = SuperPoint(max_num_keypoints=None).eval().to(self.device)
            matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(self.device)

        elif self.matching_method == "superglue":
            # Define SuperGlue settings
            superglue_config = {
                'superpoint': {
                    'nms_radius': 3,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': -1  # -1 indicates keeping all keypoints
                },
                'superglue': {
                    'weights': "outdoor",  # Can be "indoor" or "outdoor"
                    'sinkhorn_iterations': 20,
                    'match_threshold': 0.2,
                }
            }

            # Initialize the matcher for SuperGlue with the specified configuration
            matcher = Matching(superglue_config).eval().to(self.device)

            # Feature extractor is not required for SuperGlue
            extractor = None
        else:
            # Raise an error if an unsupported matching method is provided
            raise ValueError(f"Unrecognized matching method ({self.matching_method})")

        return matcher, extractor

    def _mask_tie_points(self, tps: np.ndarray, conf: np.ndarray,
                         mask1: Optional[np.ndarray],
                         mask2: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filters tie points and their confidences based on given mask arrays. 1 means no filter and
        0 means filtering

        Args:
            tps: A numpy array of tie points with shape (N, 4), where N is the number of tie points.
            conf: A numpy array of confidence values associated with each tie point.
            mask1: An optional numpy array representing the first mask to apply to the tie points.
                None if no mask is applied.
            mask2: An optional numpy array representing the second mask to apply to the tie points.
                None if no mask is applied.

        Returns:
            A tuple containing the filtered tie points and their confidences.

        Note:
            This function assumes `tps` has a structure where each row is [x1, y1, x2, y2],
                and `mask1` applies to (x1, y1) while `mask2` applies to (x2, y2).
        """

        # if both masks are None, return the inputs directly
        if mask1 is None and mask2 is None:
            return tps, conf

        num_original_tps = tps.shape[0]

        filter_values_1 = [1 if mask1 is None else mask1[int(row[1]), int(row[0])] for row in tps]
        filter_values_2 = [1 if mask2 is None else mask2[int(row[3]), int(row[2])] for row in tps]

        # Convert to numpy arrays for vectorized operations
        filter_values_1 = np.asarray(filter_values_1)
        filter_values_2 = np.asarray(filter_values_2)

        # Determine indices to keep (logical OR to find any zeros, then invert)
        keep_indices = np.logical_not(np.logical_or(filter_values_1 == 0, filter_values_2 == 0))

        # Filter the tie points and confidences
        filtered_tps = tps[keep_indices]
        filtered_conf = conf[keep_indices]

        # Optional: Print number of filtered tie points, assuming a print function and verbosity check are available
        self.logger.print(f"{num_original_tps - filtered_tps.shape[0]} of {num_original_tps} tie-points are masked",
                          color="OKBLUE")

        return filtered_tps, filtered_conf

    def _perform_additional_matching(self, img1: np.ndarray, img2: np.ndarray,
                                     input_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs additional matching by analyzing the images in original size. The bigger of the two
        images is tiled. For each tile the tie-points in this tile are analyzed and a tile with the extent
        of the equivalent tie-points in the other image is created. For these tiles tie-point matching is applied.

        Args:
            img1 (np.ndarray): The first input image.
            img2 (np.ndarray): The second input image.
            input_pts (np.ndarray): Initial set of matching points between the images. It is
                                    expected to be an array of shape (N, 4), where each row contains
                                    [x1, y1, x2, y2] coordinates of matching points in img1 and img2.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - np.ndarray: An array of refined and additional matching points between the two images,
                              with each point represented as [x1, y1, x2, y2].
                - np.ndarray: An array of confidence scores associated with each matching point.

        Note:
            The function divides the larger image into tiles and attempts to find additional
            matching points within each tile. This is particularly useful for processing large images
            or enhancing the spatial distribution of matching points across the images.
        """

        self.logger.print("Start additional matching", color="OKBLUE")

        # Determine which image is larger
        size1, size2 = np.prod(img1.shape[:2]), np.prod(img2.shape[:2])
        if size1 >= size2:
            switch_images = False
            base_img = img1
            other_img = img2
            pts_base = input_pts[:, 0:2]
            pts_other = input_pts[:, 2:4]
        else:
            switch_images = True
            base_img = img2
            other_img = img1
            pts_base = input_pts[:, 2:4]
            pts_other = input_pts[:, 0:2]

        # Calculate the number of tiles in each dimension
        num_x, num_y = (math.ceil(base_img.shape[i] / max_dim) for
                        i, max_dim in enumerate([self.max_width, self.max_height]))
        reduce_x, reduce_y = ((max_dim * num - base_img.shape[i]) // max(1, num - 1) for i, (num, max_dim) in
                              enumerate(zip([num_x, num_y], [self.max_width, self.max_height])))

        # initialize the arrays for pts and conf
        pts_additional = np.zeros([0, 4])
        conf_additional = np.array([])

        # initialize the counters
        max_counter = math.ceil(num_y) * math.ceil(num_x)
        tile_counter = 0

        # iterate all tiles
        for y_counter in range(0, math.ceil(num_y)):
            for x_counter in range(0, math.ceil(num_x)):

                tile_counter += 1  # noqa
                self.logger.print(f" Additional matching: {tile_counter}/{max_counter}", color="OKBLUE")

                # calculate the extent of the current base tile
                min_base_tile_x = x_counter * self.max_width - reduce_x * x_counter
                max_base_tile_x = (x_counter + 1) * self.max_width - reduce_x * x_counter
                min_base_tile_y = y_counter * self.max_height - reduce_y * y_counter
                max_base_tile_y = (y_counter + 1) * self.max_height - reduce_y * y_counter

                # find tie-points in the current base tile
                base_indices = np.where((pts_base[:, 0] >= min_base_tile_x) & (pts_base[:, 0] <= max_base_tile_x) &
                                        (pts_base[:, 1] >= min_base_tile_y) & (pts_base[:, 1] <= max_base_tile_y))
                base_points = pts_base[base_indices]

                if len(base_points) == 0:
                    self.logger.print("  No tie-points in base tile")
                    continue

                # get the equivalent tie-points of the other tile
                other_points = pts_other[base_indices]

                # initial step for percentiles
                step = 0

                # calculate extent of the other tile
                while True:

                    # get small and large percentile of points
                    other_percentile_sm = np.percentile(other_points, step, axis=0).astype(int)
                    other_percentile_la = np.percentile(other_points, 100 - step, axis=0).astype(int)

                    # calculate subset based on percentiles
                    min_other_tile_x = other_percentile_sm[0]  # - int(max_width / 2)
                    max_other_tile_x = other_percentile_la[0]  # + int(max_width / 2)
                    min_other_tile_y = other_percentile_sm[1]  # - int(max_height / 2)
                    max_other_tile_y = other_percentile_la[1]  # + int(max_height / 2)

                    # if extent is smaller than what we could -> enlarge subset at all ends
                    if max_other_tile_x - min_other_tile_x < self.max_width:
                        missing_width = self.max_width - (max_other_tile_x - min_other_tile_x)
                        min_other_tile_x = min_other_tile_x - int(missing_width / 2)
                        max_other_tile_x = max_other_tile_x + int(missing_width / 2)
                    if max_other_tile_y - min_other_tile_y < self.max_height:
                        missing_height = self.max_height - (max_other_tile_y - min_other_tile_y)
                        min_other_tile_y = min_other_tile_y - int(missing_height / 2)
                        max_other_tile_y = max_other_tile_y + int(missing_height / 2)

                    # last check for correct image bounds
                    min_other_tile_x = max(0, min_other_tile_x)
                    min_other_tile_y = max(0, min_other_tile_y)
                    max_other_tile_x = min(other_img.shape[1], max_other_tile_x)
                    max_other_tile_y = min(other_img.shape[0], max_other_tile_y)

                    # if tile is good -> stop loop
                    if max_other_tile_x - min_other_tile_x <= self.max_width and \
                            max_other_tile_y - min_other_tile_y <= self.max_height:
                        break

                    # if current step returns too many tie-points limit percentiles and look again
                    step = step + 5

                # extract the tiles from the images
                base_tile = base_img[min_base_tile_y:max_base_tile_y, min_base_tile_x:max_base_tile_x]
                other_tile = other_img[min_other_tile_y:max_other_tile_y, min_other_tile_x:max_other_tile_x]

                # extract tie-points for the tile
                pts_tile_1, pts_tile_2, conf_tile = self._perform_one_match(base_tile, other_tile)  # noqa

                # merge the tie points
                pts_tile = np.concatenate((pts_tile_1, pts_tile_2), axis=1)

                # adapt the tie points of the other image to account that we look at a subset
                pts_tile[:, 0] = pts_tile[:, 0] + min_base_tile_x
                pts_tile[:, 1] = pts_tile[:, 1] + min_base_tile_y
                pts_tile[:, 2] = pts_tile[:, 2] + min_other_tile_x
                pts_tile[:, 3] = pts_tile[:, 3] + min_other_tile_y

                self.logger.print(f"  {pts_tile.shape[0]} tie-points found in tile")

                # add the tie points and confidence values to the list
                pts_additional = np.concatenate((pts_additional, pts_tile), axis=0)
                conf_additional = np.concatenate((conf_additional, conf_tile))

        # if we switched the images we need to switch the tie points
        if switch_images:
            pts_additional = np.column_stack((pts_additional[:, 2], pts_additional[:, 3],
                                              pts_additional[:, 0], pts_additional[:, 1]))

        # convert tie-points to int
        pts_additional = pts_additional.astype(int)

        self.logger.print(f"{len(conf_additional)} additional matches found ({np.round(np.mean(conf_additional), 3)})")

        return pts_additional, conf_additional

    def _perform_extra_matching(self, img1: np.ndarray, img2: np.ndarray,
                                mask1: Optional[np.ndarray], mask2: Optional[np.ndarray],
                                input_pts: np.ndarray, input_conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs extra matching on two input images. A transform matrix is calculated using already existing tie-points.
        This transformation-matrix is used to calculate a possible position of one tile in the other image. Matching is
        applied on these tiles, and then the results are merged.
        Args:
            img1 (np.ndarray): The first input image.
            img2 (np.ndarray): The second input image.
            mask1 (Optional[np.ndarray]): An optional mask for the first image to exclude certain areas from matching.
            mask2 (Optional[np.ndarray]): An optional mask for the second image to exclude certain areas from matching.
            input_pts (np.ndarray): Array of initial matching points between the two images.
            input_conf (np.ndarray): Array of confidence scores for the initial matching points.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the extra matched points between the two images
                and their corresponding confidence scores. The matched points are returned as an array with
                shape (N, 4), where N is the number of matches and each row consists of (x1, y1, x2, y2) coordinates.
                The confidence scores are returned as an array with shape (N,).
        """

        self.logger.print("Start extra matching")

        # Determine which image is larger
        size1, size2 = np.prod(img1.shape[:2]), np.prod(img2.shape[:2])
        if size1 >= size2:
            switch_images = False
            base_img = img1
            other_img = img2
            other_mask = mask2
        else:
            switch_images = True
            base_img = img2
            other_img = img1
            other_mask = mask1

        # Calculate the number of tiles in each dimension
        num_x, num_y = (math.ceil(base_img.shape[i] / max_dim) for i, max_dim in
                        enumerate([self.max_width, self.max_height]))
        reduce_x, reduce_y = ((max_dim * num - base_img.shape[i]) // max(1, num - 1) for i, (num, max_dim) in
                              enumerate(zip([num_x, num_y], [self.max_width, self.max_height])))

        # only the best tie-point should be used for the transformation
        min_confidence = 0.99
        while True:
            best_pts = input_pts[input_conf > min_confidence]
            best_conf = input_conf[input_conf > min_confidence]
            if best_pts.shape[0] < 3:
                min_confidence = min_confidence - 0.05
            else:
                break

        # reduce number of points to make transformation calculation easier
        best_pts = self._select_spatial_random_points(best_pts, best_conf, base_img.shape)  # noqa

        # get affine transformation
        if switch_images is False:
            trans_mat = tf.estimate_transform('affine', best_pts[:, 0:2], best_pts[:, 2:4])
        else:
            trans_mat = tf.estimate_transform('affine', best_pts[:, 2:4], best_pts[:, 0:2])

        trans_mat = np.array(trans_mat)[0:2, :]  # noqa

        # initialize the arrays for pts and conf
        pts_extra = np.zeros([0, 4])
        conf_extra = np.array([])

        # initialize the counters
        max_counter = math.ceil(num_y) * math.ceil(num_x)
        tile_counter = 0

        # iterate all tiles
        for y_counter in range(0, math.ceil(num_y)):
            for x_counter in range(0, math.ceil(num_x)):

                tile_counter += 1  # noqa
                self.logger.print(f" Extra matching: {tile_counter}/{max_counter}")

                # calculate the extent of the current base tile
                min_base_tile_x = x_counter * self.max_width - reduce_x * x_counter
                max_base_tile_x = (x_counter + 1) * self.max_width - reduce_x * x_counter
                min_base_tile_y = y_counter * self.max_height - reduce_y * y_counter
                max_base_tile_y = (y_counter + 1) * self.max_height - reduce_y * y_counter

                # create bounding box from extent
                extent_points = np.asarray([  # noqa
                    [min_base_tile_x, min_base_tile_y],
                    [min_base_tile_x, max_base_tile_y],
                    [max_base_tile_x, max_base_tile_y],
                    [max_base_tile_x, min_base_tile_y]
                ])

                extent_points = np.hstack([extent_points, np.ones((extent_points.shape[0], 1))])
                transformed_points = np.dot(extent_points, trans_mat.T)[:, :2].astype(int)

                # get bounding box from resampled points
                min_other_tile_x = np.amin(transformed_points[:, 0])
                max_other_tile_x = np.amax(transformed_points[:, 0])
                min_other_tile_y = np.amin(transformed_points[:, 1])
                max_other_tile_y = np.amax(transformed_points[:, 1])

                # check range of bounding box
                if (min_other_tile_x < 0 and max_other_tile_x < 0) or (min_other_tile_y < 0 and max_other_tile_y < 0):
                    self.logger.print("  Skip tile (x or y below zero)")
                    continue
                if (min_other_tile_x > other_img.shape[1] and max_other_tile_x > other_img.shape[1]) or \
                        (min_other_tile_y > other_img.shape[0] and max_other_tile_y > other_img.shape[0]):
                    self.logger.print("  Skip tile (x or y over image shape)")
                    continue
                if (max_other_tile_x - min_other_tile_x < 10) or (max_other_tile_y - min_other_tile_y < 10):
                    self.logger.print("  Skip tile (tile too small)")
                    continue

                # fit bounding box to image
                min_other_tile_x = max([0, min_other_tile_x])
                min_other_tile_y = max([0, min_other_tile_y])
                max_other_tile_x = min([other_img.shape[1], max_other_tile_x])
                max_other_tile_y = min([other_img.shape[0], max_other_tile_y])

                # extract the tiles from the images
                base_tile = base_img[min_base_tile_y:max_base_tile_y, min_base_tile_x:max_base_tile_x]
                other_tile = other_img[min_other_tile_y:max_other_tile_y, min_other_tile_x:max_other_tile_x]

                if other_mask is not None:
                    other_mask_tile = other_mask[min_other_tile_y:max_other_tile_y, min_other_tile_x:max_other_tile_x]
                    if np.sum(other_mask_tile) == 0:  # noqa
                        self.logger.print("  Skip tile (tile is masked)")
                        continue

                # extract tie-points for the tile
                pts_tile_1, pts_tile_2, conf_tile = self._perform_one_match(base_tile, other_tile)  # noqa

                # merge the tie points
                pts_tile = np.concatenate((pts_tile_1, pts_tile_2), axis=1)

                # adapt the tie points of the other image to account that we look at a subset
                pts_tile[:, 0] = pts_tile[:, 0] + min_base_tile_x
                pts_tile[:, 1] = pts_tile[:, 1] + min_base_tile_y
                pts_tile[:, 2] = pts_tile[:, 2] + min_other_tile_x
                pts_tile[:, 3] = pts_tile[:, 3] + min_other_tile_y

                self.logger.print(f"  {pts_tile.shape[0]} tie-points found in tile")

                # add the tie points and confidence values to the list
                pts_extra = np.concatenate((pts_extra, pts_tile), axis=0)
                conf_extra = np.concatenate((conf_extra, conf_tile))

        # if we switched the images we need to switch the tie points
        if switch_images:
            pts_extra = np.column_stack((pts_extra[:, 2], pts_extra[:, 3], pts_extra[:, 0], pts_extra[:, 1]))

        # convert tie-points to int
        pts_extra = pts_extra.astype(int)

        self.logger.print(f"{len(conf_extra)} extra matches found ({np.round(np.mean(conf_extra), 3)})")

        return pts_extra, conf_extra

    def _perform_initial_matching(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform initial matching between two images, including an initial resizing if necessary and
        handling out-of-memory errors by retrying with smaller images.

        Args:
            img1 (np.ndarray): First image to match.
            img2 (np.ndarray): Second image to match.

        Returns:
            Tuple[np.ndarray, np.ndarray, list]: Matching points in the first image,
                matching points in the second image, and a confidence score for each match.
        """

        self.logger.print("Start initial matching")

        # deep copy of images to not change them
        img1_resized = copy.deepcopy(img1)
        img2_resized = copy.deepcopy(img2)

        # initial resize factor
        resize_factor1 = 1
        resize_factor2 = 1

        # initial resizing
        if (img1_resized.shape[0] > self.max_height) or (img1_resized.shape[1] > self.max_width):
            # calculate resize factor to decrease image to maximum allowed height or width
            resize_factor1 = min(self.max_height / img1_resized.shape[0], self.max_width / img1_resized.shape[1])

            # resize the image
            img1_resized = ri.resize_image(img1_resized, resize_factor1, "proportion")

            self.logger.print(f"Image 1 resized to {img1_resized.shape}")

        if (img2_resized.shape[0] > self.max_height) or (img2_resized.shape[1] > self.max_width):
            # calculate resize factor to decrease image to maximum allowed height or width
            resize_factor2 = min(self.max_height / img2_resized.shape[0], self.max_width / img2_resized.shape[1])

            # resize the image
            img2_resized = ri.resize_image(img2_resized, resize_factor2, "proportion")

            self.logger.print(f"Image 2 resized to {img1_resized.shape}")

        pts0, pts1, conf = self._perform_one_match(img1_resized, img2_resized)

        # merge the tie-points of left and right image
        pts = np.concatenate((pts0, pts1), axis=1)

        # adapt the tie points to account for resizing
        pts[:, 0] = pts[:, 0] * 1 / resize_factor1  # noqa
        pts[:, 1] = pts[:, 1] * 1 / resize_factor1
        pts[:, 2] = pts[:, 2] * 1 / resize_factor2
        pts[:, 3] = pts[:, 3] * 1 / resize_factor2
        pts = pts.astype(int)

        self.logger.print(f"{len(conf)} initial matches found ({np.round(np.mean(conf), 3)})")

        return pts, np.array(conf)

    def _perform_one_match(self, input_img1: np.ndarray,
                           input_img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs feature matching between two input images using either "lightglue" or "superglue"
        matching methods. It returns the matching keypoints in both images along with their
        confidence scores. The function handles GPU out-of-memory errors by iteratively reducing
        the image size until the matching can be performed successfully.

        Args:
            input_img1 (np.ndarray): The first input image as a NumPy array.
            input_img2 (np.ndarray): The second input image as a NumPy array.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three elements:
                - np.ndarray: Matched keypoints in the first image, with shape (N, 2).
                - np.ndarray: Matched keypoints in the second image, with shape (N, 2).
                - np.ndarray: Confidence scores of the matches, with shape (N,).

        Raises:
            ValueError: If an unrecognized matching method is specified.

        Note:
            The function attempts to manage GPU memory usage by reducing image sizes if a
            RuntimeError indicating out-of-memory (OOM) occurs. This resizing process continues
            until matching can be completed or further resizing is not feasible. Matched points
            are adjusted to account for any resizing that occurs.
        """

        # Deep copy images to avoid modifying the original data
        sg_img1 = copy.deepcopy(input_img1)
        sg_img2 = copy.deepcopy(input_img2)

        # get width and height of the images
        img_width1, img_height1 = input_img1.shape[1], input_img2.shape[0]
        img_width2, img_height2 = input_img2.shape[1], input_img2.shape[0]

        # init resize factors
        resize_factor1 = 1
        resize_factor2 = 1

        # try as long we get a tie-point detection
        while True:

            # try to detect tie-points
            try:

                # Convert images to PyTorch tensors and normalize
                sg_img1 = torch.from_numpy(sg_img1)[None, None].float() / 255.0
                sg_img2 = torch.from_numpy(sg_img2)[None, None].float() / 255.0

                # Move images to the specified device
                sg_img1 = sg_img1.to(self.device)
                sg_img2 = sg_img2.to(self.device)

                if self.matching_method == "lightglue":
                    # Extract features from both images
                    feats0 = self.extractor.extract(sg_img1)  # noqa
                    feats1 = self.extractor.extract(sg_img2)  # noqa

                    # Match features between the two images
                    matches01 = self.matcher({'image0': feats0, 'image1': feats1})  # noqa
                    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

                    # Extract keypoints and matches, removing the batch dimension
                    kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
                    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

                    # Convert keypoints and confidence scores to NumPy arrays
                    pts0 = m_kpts0.cpu().numpy()
                    pts1 = m_kpts1.cpu().numpy()
                    conf = matches01['scores'].detach().cpu().numpy()

                elif self.matching_method == "superglue":
                    # Specify the data to be extracted
                    keys = ['keypoints', 'scores', 'descriptors']

                    # Detect features in the first image
                    last_data = self.matcher.superpoint({'image': sg_img1})  # noqa
                    last_data = {k + '0': last_data[k] for k in keys}
                    last_data["image0"] = sg_img1

                    # Match features between the first and second images
                    pred = self.matcher({**last_data, 'image1': sg_img2})  # noqa

                    # Extract keypoints, matches, and confidence scores
                    kpts0 = last_data['keypoints0'][0].cpu().numpy()
                    kpts1 = pred['keypoints1'][0].cpu().numpy()
                    matches_superglue = pred['matches0'][0].cpu().numpy()
                    confidence = pred['matching_scores0'][0].detach().cpu().numpy()

                    # Filter valid matching keypoints
                    valid = matches_superglue > -1
                    pts0 = kpts0[valid]
                    pts1 = kpts1[matches_superglue[valid]]
                    conf = confidence[valid]

                else:
                    raise ValueError(f"Unrecognized matching method ({self.matching_method})")

                # matching was successful, so break the loop
                break

            # catch error and check for oom
            except RuntimeError as e:

                print(e)

                if "out of memory" in str(e) or "OutOfMemoryError" in str(e):
                    # free the gpu
                    torch.cuda.empty_cache()

                    # calculate new height and width
                    img_width1 = int(OOM_REDUCE_VALUE * img_width1)
                    img_height1 = int(OOM_REDUCE_VALUE * img_height1)

                    img_width2 = int(OOM_REDUCE_VALUE * img_width2)
                    img_height2 = int(OOM_REDUCE_VALUE * img_height2)

                    resize_factor1 = OOM_REDUCE_VALUE * resize_factor1
                    resize_factor2 = OOM_REDUCE_VALUE * resize_factor2

                    sg_img1 = ri.resize_image(input_img1, (img_height1, img_width1), "size")  # noqa
                    sg_img2 = ri.resize_image(input_img2, (img_height2, img_width2), "size")  # noqa

                else:
                    raise e

        # adapt the tie points to account for resizing
        pts0[:, 0] = pts0[:, 0] * 1 / resize_factor1
        pts0[:, 1] = pts0[:, 1] * 1 / resize_factor1
        pts1[:, 0] = pts1[:, 0] * 1 / resize_factor2
        pts1[:, 1] = pts1[:, 1] * 1 / resize_factor2

        pts0 = pts0.astype(int)
        pts1 = pts1.astype(int)

        return pts0, pts1, conf

    @staticmethod
    def _prepare_images(input_img1: np.ndarray, input_img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares two images by making a deep copy and converting them to grayscale. This function
        handles both (height, width, channels) and (channels, height, width) formats by checking
        the shape of the images and adjusting accordingly.

        Args:
            input_img1 (np.ndarray): The first input image. Can be grayscale, BGR, or channel-first format.
            input_img2 (np.ndarray): The second input image. Can be grayscale, BGR, or channel-first format.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the two processed images in grayscale.

        Note:
            This function assumes that the input images are either in grayscale, BGR format,
            or in channel-first format. It automatically adjusts channel-first images to
            channel-last before converting to grayscale.
        """

        # Deep copy to not change the original images
        img1 = copy.deepcopy(input_img1)
        img2 = copy.deepcopy(input_img2)

        # Convert images from (channels, height, width) to (height, width, channels) if necessary
        if img1.ndim == 3 and img1.shape[0] == 3:
            img1 = np.moveaxis(img1, 0, -1)
        if img2.ndim == 3 and img2.shape[0] == 3:
            img2 = np.moveaxis(img2, 0, -1)

        # Convert images to grayscale if they are not already
        if img1.ndim == 3 and img1.shape[-1] == 3:  # Check if the image has 3 channels at the last dimension
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if img2.ndim == 3 and img2.shape[-1] == 3:  # Check if the image has 3 channels at the last dimension
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        return img1, img2

    def _select_spatial_random_points(self, points: np.ndarray, confidences: np.ndarray,
                                      img_dims: Tuple[int, int]) -> np.ndarray:
        """
        Selects spatially distributed points based on confidence scores and image dimensions.

        This method divides the image into a grid and attempts to select points from as much
        different grid cells as possible, prioritizing points with higher confidence scores.
        If not enough unique cells can be filled to reach the desired number of points,
        it back-fills with the highest confidence points not yet selected.

        Args:
            points (np.ndarray): The array of points to select from, shape (N, 2) or (N, 4).
            confidences (np.ndarray): The array of confidence scores for each point, shape (N,).
            img_dims (Tuple[int, int]): The dimensions of the image as (width, height).

        Returns:
            np.ndarray: The selected points based on spatial distribution and confidence,
                shape (num_transform_points, 2) or (num_transform_points, 4).

        Note:
            If the number of available points is less than `num_transform_points`, all points are returned.
        """

        # Check if points array is empty or has fewer points than requested
        if points.size == 0 or len(points) < self.num_transform_points:
            return points

        # Determine the grid size based on the number of points to select
        grid_size = int(np.ceil(np.sqrt(self.num_transform_points)))

        # Calculate the size of each grid cell
        cell_size = (img_dims[0] / grid_size, img_dims[1] / grid_size)

        selected_indices = []
        cells_occupied = set()

        # Sort indices by confidence to prioritize selection
        sorted_indices = np.argsort(-confidences)

        for idx in sorted_indices:
            if len(selected_indices) >= self.num_transform_points:
                break

            point = points[idx]

            # Determine the grid cell for the current point
            cell_x = int(point[0] // cell_size[0])
            cell_y = int(point[1] // cell_size[1])
            cell_id = (cell_x, cell_y)

            # Check if this cell has already been used
            if cell_id not in cells_occupied:
                selected_indices.append(idx)
                cells_occupied.add(cell_id)

        # If not enough points were selected, fill in with remaining highest confidence points
        if len(selected_indices) < self.num_transform_points:
            remaining_indices = [i for i in sorted_indices if i not in selected_indices]
            selected_indices.extend(remaining_indices[:self.num_transform_points - len(selected_indices)])

        # Select points and confidences based on the final indices
        selected_points = points[selected_indices]

        return selected_points
