# Package imports
import cv2
import numpy as np
from scipy.ndimage import zoom
from shapely.geometry import Polygon

# Custom imports
import src.base.find_tie_points as ftp
import src.georef.snippets.apply_transform as at
import src.georef.snippets.calc_transform as ct
import src.load.load_satellite as ls


def georef_ortho(ortho, footprints, save_path=None):

    # Get the min and max for all footprints
    min_x = min([footprint.bounds[0] for footprint in footprints])
    min_y = min([footprint.bounds[1] for footprint in footprints])
    max_x = max([footprint.bounds[2] for footprint in footprints])
    max_y = max([footprint.bounds[3] for footprint in footprints])

    # Create the bounding rectangle
    bounding_rectangle = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])

    # get satellite image for the bounding rectangle
    sat, sat_transform = ls.load_satellite(bounding_rectangle)

    # resize ortho to sat size
    zoom_factors = (sat.shape[1] / ortho.shape[0], sat.shape[2] / ortho.shape[1])
    ortho_resized = zoom(ortho, zoom_factors, order=3)

    # find tps between ortho and sat
    tpd = ftp.TiePointDetector('lightglue')
    points, _ = tpd.find_tie_points(sat, ortho_resized)

    # remove outliers
    _, filtered = cv2.findHomography(points[:, 0:2], points[:, 2:4], cv2.RANSAC, 5.0)
    filtered = filtered.flatten()
    points = points[filtered == 0]

    # convert points to absolute values
    absolute_points = np.array([sat_transform * tuple(point) for point in points[:, 0:2]])
    points[:, 0:2] = absolute_points

    # account for resizing
    points[:, 2] = points[:, 2] * (1 / zoom_factors[1])
    points[:, 3] = points[:, 3] * (1 / zoom_factors[0])

    # calculate the transform
    transform, residuals = ct.calc_transform(ortho, points,
                                             transform_method="rasterio",
                                             gdal_order=3)

    if save_path is not None:
        at.apply_transform(ortho, transform, save_path)

    transform = np.reshape(transform, (3, 3))

    return transform