import sys
import xml.etree.ElementTree as ET
import numpy as np

from skimage import transform as tf

debug_show_transformed_points = False

def resample_tie_points(project_folder, img_id, points, scan_res = 0.025, catch=True):

    # load the image xml
    try:
        image_xml = ET.parse(project_folder + f"/Ori-InterneScan/MeasuresIm-{img_id}.tif.xml")
        img_root = image_xml.getroot()
    except:
        if catch:
            return None
        else:
            raise ValueError

    # load the image points in np arr
    points_image = []
    for elem in img_root[0][1:]:
        val = list(elem)[1].text
        x = int(val.split(" ")[0])
        y = int(val.split(" ")[1])
        points_image.append([x,y])
    points_image = np.array(points_image)

    # load the camera xml
    cam_xml = ET.parse(project_folder + f"/Ori-InterneScan/MeasuresCamera.xml")
    cam_root = cam_xml.getroot()

    # load the camera points in np arr
    points_camera = []
    for elem in cam_root[1:]:
        val = list(elem)[1].text
        x = float(val.split(" ")[0])
        y = float(val.split(" ")[1])
        points_camera.append([x,y])
    points_camera = np.array(points_camera)

    # adapt cam points to image
    points_camera = np.round(points_camera / scan_res).astype(int)

    # get min and max of camera points for x and y
    min_x, max_x = sys.maxsize, 0
    min_y, max_y = sys.maxsize, 0

    for i, elem in enumerate(points):
        if elem[0] < min_x:
            min_x = elem[0]
        if elem[0] > max_x:
            max_x = elem[0]

        if elem[1] < min_y:
            min_y = elem[1]
        if elem[1] > max_y:
            max_y = elem[1]

    # get affine transformation
    trans_mat = tf.estimate_transform('affine', points_image, points_camera)
    trans_mat = np.array(trans_mat)[0:2, :]

    # apply transformation to the points
    points_tr = np.empty_like(points)

    points_tr[:,0] = trans_mat[0][0] * points[:,0] + trans_mat[0][1] * points[:,1] + trans_mat[0][2]
    points_tr[:,1] = trans_mat[1][0] * points[:,0] + trans_mat[1][1] * points[:,1] + trans_mat[1][2]

    #no negative values allowed
    if (points < 0).sum() != 0:
        p.print_v(f"there were invalid points (<0) before resampling in {img_id}", color="red")
        exit()

    # if (points_tr < 0).sum() != 0:
    #     p.print_v(f"there are invalid tr_points (<0) after resampling in {img_id}", color="red")
    #     exit()


    if debug_show_transformed_points:

        img_path = project_folder + "/images_orig/"
        tr_path = project_folder

        img = liff.load_image_from_file(img_id, image_path=img_path, catch=False)
        tr = liff.load_image_from_file("OIS-Reech_" + img_id, image_path=tr_path, catch=False)

        points_all = list(zip(points[:,0], points[:,1], points_tr[:,0], points_tr[:,1]))

        dt.display_tiepoints([img, tr], points_all, reduce_points=True, num_points=1000, title=img_id)

    return points_tr

if __name__ == "__main__":

    project_folder = "/data_1/ATM/data/sfm/" \
                     "projects/final_test"

    resample_tie_points()