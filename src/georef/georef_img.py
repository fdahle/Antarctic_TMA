import numpy as np

import src.base.find_tie_points as ftp


class GeorefImage:

    def __init__(self, enhance_image=True):

        self.enhance_image = True

        # initialize tie point detector
        self.tp_finder = ftp.TiePointDetector(matching_method="lightglue")

    def georeference(self, input_image, georeferenced_images, georeferenced_transforms,
                     mask=None, georeferenced_masks=None):

        all_tps = np.empty([0, 4])
        all_confs = np.empty([0, 1])

        # iterate all geo-referenced images
        for i, georeferenced_image in enumerate(georeferenced_images):

            # get the correct transform
            georeferenced_transform = georeferenced_transforms[i]

            # get the correct mask
            if georeferenced_masks is not None:
                georeferenced_mask = georeferenced_masks[i]
            else:
                georeferenced_mask = None

            # apply tie-point matching between geo-referenced image and input image
            tps, conf = self.tp_finder.find_tie_points(georeferenced_image, input_image,
                                                       georeferenced_mask, mask)

            # convert tie-points to absolute values
            absolute_points = np.array([georeferenced_transform * tuple(point) for point in tps[:, 0:2]])
            tps[:, 0:2] = absolute_points
