import cv2
import numpy as np

import src.base.find_tie_points as ftp
import src.load.load_image as li

import src.display.display_images as di


def fine_georef(image, transform):
    path_img = "/data_1/ATM/data_1/aerial/icebridge/DMS_1381716_00183_20121028_16160486.tif"

    # remove 0, 0, 1 from the transform matrix
    transform = np.asarray(transform)
    transform = transform[:6]
    transform = transform.reshape([2, 3])

    cos_theta = transform[0, 0]
    sin_theta = transform[1, 0]

    angle = np.degrees(np.arctan2(sin_theta, cos_theta))

    import src.base.rotate_image as ri
    image, _ = ri.rotate_image(image, angle)

    print(image.shape)

    # get the accurate geo-referenced image
    image_accurate = li.load_image(path_img)

    di.display_images(image)

    detector = ftp.TiePointDetector(matching_method="lightglue")

    # find tie points between the two images
    tie_points, conf = detector.find_tie_points(image, image_accurate)

    di.display_images([image, image_accurate], tie_points=tie_points, tie_points_conf=conf)

    print(tie_points.shape)
