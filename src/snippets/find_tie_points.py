import copy
import cv2
import numpy as np
import torch

from typing import Tuple, Optional, Dict, Any

import snippets.resize_image as ri

from external.lightglue import LightGlue, SuperPoint
from external.lightglue.utils import rbd
from external.SuperGlue.matching import Matching

OOM_REDUCE_VALUE = 0.9

class TiePointDetector:

    def __init__(self, matching_method: str, verbose):
        self.matching_method = matching_method.lower()
        self.verbose = verbose

        self.max_height = 2000
        self.max_width = 2000

        # Initialize device based on CUDA availability
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

        # init matcher
        self.matcher, self.extractor = self._init_matcher()

        print(f"TiePointDetector initialized using {matching_method} on {self.device}")


    def find_tie_points(self, input_img1, input_img2):

        img1, img2 = self._prepare_images(input_img1, input_img2)

        self._perform_initial_matching(img1, img2)

    def _apply_mask_filtering(self):
        pass


    def _filter_outliers(self):
        pass


    def _init_matcher(self):
        """
        Initializes and returns the matching algorithm and the optional feature extractor based on the specified method.

        Args:
            matching_method (str): The name of the matching method to initialize. Supported methods: "LightGlue",
                "SuperGlue".
            device (str): The device to use for the computations. Typically "cpu" or "cuda".

        Returns:
            Tuple[Any, Optional[Any]]: A tuple containing the matcher algorithm and an optional feature extractor.
            The feature extractor is None if not required by the matching method.

        Raises:
            ValueError: If an unrecognized matching method is specified.
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


    def _perform_additional_matching(self):
        pass

    def _perform_extra_matching(self):
        pass

    def _perform_initial_matching(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
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

        img1_resized = copy.deepcopy(img1)
        img2_resized = copy.deepcopy(img2)

        resize_factor1 = 1
        resize_factor2 = 1

        # initial resizing
        if (img1_resized.shape[0] > self.max_height) or (img1_resized.shape[1] > self.max_width):

            # calculate resize factor to decrease image to maximum allowed height or width
            resize_factor1 = min(self.max_height / img1_resized.shape[0], self.max_width / img1_resized.shape[1])

            # resize the image
            img1_resized = ri.resize_image(img1_resized, resize_factor1, "proportion")

            print(f"Image 1 resized to {img1_resized.shape}")

        if (img2_resized.shape[0] > self.max_height) or (img2_resized.shape[1]> self.max_width):

            # calculate resize factor to decrease image to maximum allowed height or width
            resize_factor2 = min(self.max_height / img2_resized.shape[0], self.max_width / img2_resized.shape[1])

            # resize the image
            img2_resized = ri.resize_image(img2_resized, resize_factor2, "proportion")

            print(f"Image 2 resized to {img1_resized.shape}")


        print("Perform initial matching")

        while True:
            try:
                pts0, pts1, conf = self._perform_one_match(img1_resized, img2_resized)
                break  # Break the loop if matching is successful

            except RuntimeError as e:

                if "out of memory" in str(e):

                    # free the gpu
                    torch.cuda.empty_cache()

                    # calculate new height and width
                    max_width = int(OOM_REDUCE_VALUE * max_width)
                    max_height = int(OOM_REDUCE_VALUE * max_height)

                    img1_resized = ri.resize_image(img1_resized, (max_height, max_width))
                    img2_resized = ri.resize_image(img2_resized, (max_height, max_width))

                    print((f"Out of memory, try again with new image size for "
                           f"img1 {img1_resized.shape} and img2 {img2_resized.shape}"))
                else:
                    raise  # Reraise the exception if it's not an out-of-memory error

        return pts0, pts1, conf

    def _perform_one_match(self, input_img1: np.ndarray, input_img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs feature matching between two input images and returns matching points and their confidence scores.

        Args:
            input_img1 (np.ndarray): The first input image as a NumPy array.
            input_img2 (np.ndarray): The second input image as a NumPy array.
            matching_method (str): The method to use for matching. Supported methods: "lightglue", "superglue".
            extractor (Any): The feature extractor model, required for "lightglue".
            matcher (Any): The matching model to use for finding correspondences between images.
            device (str): The device to use for computation ("cpu" or "cuda").

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing matched keypoints in the first image,
                matched keypoints in the second image, and their confidence scores.
        """

        # Deep copy images to avoid modifying the original data
        sg_img_1 = copy.deepcopy(input_img1)
        sg_img_2 = copy.deepcopy(input_img2)

        # Convert images to PyTorch tensors and normalize
        sg_img_1 = torch.from_numpy(sg_img_1)[None, None].float() / 255.0
        sg_img_2 = torch.from_numpy(sg_img_2)[None, None].float() / 255.0

        # Move images to the specified device
        sg_img_1 = sg_img_1.to(self.device)
        sg_img_2 = sg_img_2.to(self.device)

        if self.matching_method == "lightglue":
            # Extract features from both images
            feats0 = self.extractor.extract(sg_img_1)
            feats1 = self.extractor.extract(sg_img_2)

            # Match features between the two images
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
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
            last_data = self.matcher.superpoint({'image': sg_img_1})
            last_data = {k + '0': last_data[k] for k in keys}
            last_data["image0"] = sg_img_1

            # Match features between the first and second images
            pred = self.matcher({**last_data, 'image1': sg_img_2})

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

        return pts0, pts1, conf


    def _prepare_images(self, input_img1: np.ndarray, input_img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
