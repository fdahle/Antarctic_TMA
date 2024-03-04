import copy

import cv2
import numpy as np

from rasterio.transform import Affine
from typing import Optional, Tuple

# import base functions
import src.base.enhance_image as eh
import src.base.find_tie_points as ftp

# import display functions
import src.display.display_images as di

# import georef snippet functions
import src.georef.snippets.calc_transform as ct

debug_display_steps = True

class GeorefImage:

    def __init__(self, enhance_image: bool = True, enhance_georef_images: bool = True,
                 transform_method: str = "rasterio", transform_order: int = 3,
                 filter_outliers: bool = True) -> None:
        """
        Initialize the GeorefImage class with various settings for geo-referencing historical images
        with neighbouring, already geo-referenced historical images.
        Args:
            enhance_image (bool): Whether to enhance the input image before geo-referencing.
                Defaults to True.
            enhance_georef_images (bool): Whether to enhance the already geo-referenced images.
                Defaults to True.
            transform_method (str): The method used for transformation calculations.
                Defaults to "rasterio".
            transform_order (int): The order of transformation for geo-referencing.
                Defaults to 3.
            filter_outliers (bool): Whether to filter outliers in tie points detection.
                Defaults to True.
        """

        # settings for enhance image
        self.enhance_image = enhance_image
        self.enhance_georef_images = enhance_georef_images

        # settings for filtering
        self.filter_outliers = filter_outliers

        # initialize tie point detector
        self.tp_finder = ftp.TiePointDetector(matching_method="lightglue")

        # transform settings
        self.transform_method = transform_method
        self.transform_order = transform_order

    def georeference(self, image: np.ndarray, georeferenced_images: list[np.ndarray],
                     georeferenced_transforms: list, mask: np.ndarray = None,
                     georeferenced_masks: list[np.ndarray] = None) -> Tuple[Optional[np.ndarray],
                                                                            Optional[np.ndarray],
                                                                            Optional[np.ndarray],
                                                                            Optional[np.ndarray]]:
        """
        Performs the geo-referencing on an image using a list of neighbouring geo-referenced images
        and their transformations. Possible to filter tie-points with masks.
        Args:
            image (np.ndarray): The input image to be georeferenced.
            georeferenced_images (list[np.ndarray]): A list of geo-referenced images.
            georeferenced_transforms (list): A list of transformations corresponding to the geo-referenced
                images.
            mask (np.ndarray, optional): The mask for the input image. Defaults to None.
            georeferenced_masks (list[np.ndarray], optional): A list of masks for each georeferenced image.
                Defaults to None.
        Returns:
            transform (np.ndarray): The transformation matrix of the geo-referenced image as a NumPy array.
            residuals(np.ndarray): The residuals of the transformation points as a NumPy array.
            tps (np.ndarray): The tie points between the input image and the satellite image as a NumPy array.
            conf (np.ndarray): The confidence scores associated with the tie points as a NumPy array.
        """
        tps = np.empty([0, 4])
        conf = np.empty([0])

        if self.enhance_image:
            image = eh.enhance_image(image, mask)

        # iterate all geo-referenced images
        for i, georeferenced_image in enumerate(georeferenced_images):

            # get the correct transform
            georeferenced_transform = georeferenced_transforms[i]

            # get the correct mask
            if georeferenced_masks is not None:
                georeferenced_mask = georeferenced_masks[i]
            else:
                georeferenced_mask = None

            # enhance geo-referenced image if wished
            if self.enhance_georef_images:
                georeferenced_image = eh.enhance_image(georeferenced_image, georeferenced_mask)

            # apply tie-point matching between geo-referenced image and input image
            tps_img, conf_img = self.tp_finder.find_tie_points(georeferenced_image, image,
                                                               georeferenced_mask, mask)

            # check if georef transform is a numpy array
            if isinstance(georeferenced_transform, np.ndarray):
                a, b, c = georeferenced_transform[0]
                d, e, f = georeferenced_transform[1]

                # Create the Rasterio affine transform
                georeferenced_transform = Affine(a, b, c, d, e, f)

            # backup copy for display
            tps_display = copy.deepcopy(tps_img)

            # convert tie-points to absolute values
            absolute_points = np.array([georeferenced_transform * tuple(point) for point in tps_img[:, 0:2]])
            tps_img[:, 0:2] = absolute_points

            # last filtering of the tie-points
            if self.filter_outliers:
                _, filtered = cv2.findHomography(tps_img[:, 0:2], tps_img[:, 2:4], cv2.RANSAC, 5.0)
                filtered = filtered.flatten()

                # 1 means outlier
                tps_img = tps_img[filtered == 0]
                tps_display = tps_display[filtered == 0]
                conf_img = conf_img[filtered == 0]

                print(f"{np.count_nonzero(filtered)} outliers removed with RANSAC")

            print(f"{tps_img.shape[0]} tie points found between the input image and the {i+1}th geo-referenced image")

            if debug_display_steps:
                di.display_images([georeferenced_image, image],
                                  tie_points=tps_display, tie_points_conf=conf_img)

            # add tps and conf to global list
            tps = np.concatenate([tps, tps_img])
            conf = np.concatenate([conf, conf_img])

        # calculate the transformation-matrix
        transform, residuals = ct.calc_transform(image, tps,
                                                 transform_method=self.transform_method,
                                                 gdal_order=self.transform_order)

        return transform, residuals, tps, conf
