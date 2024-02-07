import copy

import numpy as np

import src.base.enhance_image as eh
import src.base.find_tie_points as ftp


class GeorefImage:

    def __init__(self, enhance_image=True, enhance_georef_images=True):

        self.enhance_image = True
        self.enhance_georef_images = True

        # initialize tie point detector
        self.tp_finder = ftp.TiePointDetector(matching_method="lightglue")

    def georeference(self, image, georeferenced_images, georeferenced_transforms,
                     mask=None, georeferenced_masks=None):

        all_tps = np.empty([0, 4])
        all_confs = np.empty([0, 1])

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
            tps, conf = self.tp_finder.find_tie_points(georeferenced_image, image,
                                                       georeferenced_mask, mask)

            # convert tie-points to absolute values
            absolute_points = np.array([georeferenced_transform * tuple(point) for point in tps[:, 0:2]])
            tps[:, 0:2] = absolute_points

            # add tps and conf to global list
            all_tps = np.concatenate([all_tps, tps])
            all_confs = np.concatenate([all_confs], conf)

        print(all_tps, all_confs)
