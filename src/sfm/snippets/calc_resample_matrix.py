# Package imports
import numpy as np
import xml.etree.ElementTree as Et
from skimage import transform as tf


def calc_resample_matrix(project_folder, image_id, scan_res=0.025):
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
    #points_camera = np.round(points_camera / scan_res).astype(int)

    print("IMG")
    print(points_image)
    print("CAM")
    print(points_camera)
    print("")

    # get affine transformation
    trans_mat = tf.estimate_transform('affine', points_image, points_camera)
    trans_mat = np.array(trans_mat)[0:2, :]

    return trans_mat
