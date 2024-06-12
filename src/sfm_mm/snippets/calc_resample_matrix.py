"""Calculates the resampling matrix for image in MicMac"""

# Library imports
import cv2
import numpy as np
import xml.etree.ElementTree as Et
from skimage import transform as tf

import src.display.display_images as di
import src.load.load_image as li

# Debug variables
debug_show_images = False


def calc_resample_matrix(project_folder: str, image_id: str, scan_res: float = 0.025) -> np.ndarray:
    """
    Calculates the resampling matrix for an image based on its MicMac xml-file.
    The matrix is used to resample the image based on the scan resolution.
    Note that the xml files are expected to be in the format
    "project_folder/Ori-InterneScan/MeasuresIm-{image_id}.tif.xml"
    Args:
        project_folder (str): The path to the project folder.
        image_id (str): The ID of the image.
        scan_res (float, optional): The scan resolution. Defaults to 0.025.
    Returns:
        np.ndarray: The affine transformation matrix.
    """
    # load the image xml
    image_xml = Et.parse(project_folder + f"/Ori-InterneScan/MeasuresIm-{image_id}.tif.xml")
    img_root = image_xml.getroot()

    # load the image points in np arr
    points_image = []
    for elem in img_root[0][1:]:
        val = list(elem)[1].text
        x = int(val.split(" ")[0])
        y = int(val.split(" ")[1])
        points_image.append([x, y])
    points_image = np.array(points_image)

    # load the camera xml
    cam_xml = Et.parse(project_folder + f"/Ori-InterneScan/MeasuresCamera.xml")
    cam_root = cam_xml.getroot()

    # load the camera points in np arr
    points_camera = []
    for elem in cam_root[1:]:
        val = list(elem)[1].text
        x = float(val.split(" ")[0])
        y = float(val.split(" ")[1])
        points_camera.append([x, y])
    points_camera = np.array(points_camera)

    # adapt cam points to image
    points_camera = np.round(points_camera / scan_res).astype(int)

    # get affine transformation
    trans_mat = tf.estimate_transform('affine', points_image, points_camera)
    trans_mat = np.array(trans_mat)[0:2, :]

    if debug_show_images:

        # load the images
        image = li.load_image(image_id)
        image_resampled = li.load_image(project_folder + "/OIS-Reech_" + image_id + ".tif")

        # resample the one image
        rows, cols = image.shape
        transformed_image = cv2.warpAffine(image, trans_mat, (cols, rows))

        style_config = {"title": image_id, "titles_sup": [image.shape, transformed_image.shape, image_resampled.shape]}
        di.display_images([image[:500, :500],
                           transformed_image[:500, :500],
                           image_resampled[:500, :500]], style_config=style_config)

    return trans_mat
