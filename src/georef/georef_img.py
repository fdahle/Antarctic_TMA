import cv2

import numpy as np

# import base functions
import src.base.enhance_image as eh
import src.base.find_tie_points as ftp

# import georef snippet functions
import src.georef.snippets.calc_transform as ct


class GeorefImage:

    def __init__(self, enhance_image=True, enhance_georef_images=True):

        self.enhance_image = True
        self.enhance_georef_images = True

        # initialize tie point detector
        self.tp_finder = ftp.TiePointDetector(matching_method="lightglue")

    def georeference(self, image, georeferenced_images, georeferenced_transforms,
                     mask=None, georeferenced_masks=None):

        tps = np.empty([0, 4])
        conf = np.empty([0, 1])

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

            # convert tie-points to absolute values
            absolute_points = np.array([georeferenced_transform * tuple(point) for point in tps_img[:, 0:2]])
            tps_img[:, 0:2] = absolute_points

            # add tps and conf to global list
            tps = np.concatenate([tps, tps_img])
            conf = np.concatenate([conf, conf_img])

        # last filtering of the tie-points
        if self.filter_outliers:
            _, filtered = cv2.findHomography(tps[:, 0:2], tps[:, 2:4], cv2.RANSAC, 5.0)
            filtered = filtered.flatten()

            # 1 means outlier
            tps = tps[filtered == 0]
            conf = conf[filtered == 0]

            print(f"{np.count_nonzero(filtered)} outliers removed with RANSAC")

        transform, residuals = ct.calc_transform(image, tps,
                                                 transform_method=self.transform_method,
                                                 gdal_order=self.transform_order)

        return transform, residuals, tps, conf