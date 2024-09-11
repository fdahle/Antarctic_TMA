import os
import numpy as np
import pandas as pd
import struct
import xml.etree.ElementTree as ET
from tqdm import tqdm

import src.base.find_tie_points as ftp
import src.export.export_ply as ep
import src.load.load_image as li
import src.sfm_agi.snippets.adapt_frame as af
import src.sfm_agi.snippets.zip_folder as zp

def create_custom_tie_points(chunk,
                             files_fld,
                             img_folder,
                             mask_folder=None,
                             matching_method='all', min_conf=0.7,
                             min_points_img=10, min_tps = 25, max_tps = 4000):

    # here the image ids are saved
    image_ids = []

    # get the tif-images in the folder
    for filename in os.listdir(img_folder):
        if filename.lower().endswith('.tif'):
            image_ids.append(filename)

    # here the images and masks are saved
    image_dict = {}
    mask_dict = {}

    # list with all combinations of images
    combinations = []

    if matching_method == 'all':

        # create a list with all combinations of images
        for i in range(len(image_ids)):
            for j in range(i+1, len(image_ids)):

                # skip identical images
                if image_ids[i] == image_ids[j]:
                    continue

                if (image_ids[j], image_ids[i]) in combinations:
                    continue

                # add the combination to the list
                combinations.append((image_ids[i], image_ids[j]))

    # init tie-point detector
    tpd = ftp.TiePointDetector('lightglue', verbose=True,
                               min_conf_value=min_conf)

    # here the tie points & confidence values are saved
    tp_dict = {}
    conf_dict = {}

    # iterate over all combinations
    print("Iterate combinations")
    for id_img1, id_img2 in tqdm(combinations):

        # load the images
        if id_img1 not in image_dict:
            path_img1 = os.path.join(img_folder, id_img1)
            image_dict[id_img1] = li.load_image(path_img1)
        image1 = image_dict[id_img1]

        if id_img2 not in image_dict:
            path_img2 = os.path.join(img_folder, id_img2)
            image_dict[id_img2] = li.load_image(path_img2)
        image2 = image_dict[id_img2]

        # load the masks
        if mask_folder is not None:
            if id_img1 not in mask_dict:
                path_mask1 = os.path.join(mask_folder, id_img1)
                path_mask1 = path_mask1.replace('.tif', '_mask_adapted.tif')
                mask_dict[id_img1] = li.load_image(path_mask1)
            mask1 = mask_dict[id_img1]
            if id_img2 not in mask_dict:
                path_mask2 = os.path.join(mask_folder, id_img2)
                path_mask2 = path_mask2.replace('.tif', '_mask_adapted.tif')
                mask_dict[id_img2] = li.load_image(path_mask2)
            mask2 = mask_dict[id_img2]
        else:
            mask1 = None
            mask2 = None

        # get the tie points
        tps, conf = tpd.find_tie_points(image1, image2,
                                        mask1=mask1, mask2=mask2)

        print(f"Number of tie points between {id_img1} and {id_img2}: ", tps.shape[0])

        # skip if too few tie points are found
        if tps.shape[0] < min_tps:
            continue

        # limit number of tie points
        if tps.shape[0] > max_tps:

            # select the top points
            top_indices = np.argsort(conf)[-max_tps:][::-1]
            tps = tps[top_indices]

        # save the tie points and conf
        tp_dict[(id_img1, id_img2)] = tps
        conf_dict[(id_img1, id_img2)] = conf


    # here tie points are saved per image
    points_per_image = {}

    # iterate over all tie points to get points per image
    for key, value in tqdm(tp_dict.items()):

        # get the two ids
        img1_id = key[0]
        img2_id = key[1]

        # create an empty numpy array for each image if it does not exist
        if img1_id not in points_per_image:
            points_per_image[img1_id] = np.zeros((0, 2))
        if img2_id not in points_per_image:
            points_per_image[img2_id] = np.zeros((0, 2))

        # get the points per image
        points1 = value[:, :2]
        points2 = value[:, 2:]

        # add the points to the numpy array
        points_per_image[img1_id] = np.vstack((points_per_image[img1_id], points1))
        points_per_image[img2_id] = np.vstack((points_per_image[img2_id], points2))

    # create path where the ply files are saved
    path_pc_fld = os.path.join(files_fld, "0", "0", "point_cloud")
    if not os.path.exists(path_pc_fld):
        os.makedirs(path_pc_fld)

    # generate ply files
    num_matches = _generate_ply_files(tp_dict, path_pc_fld)

    # create numpy array for the tracks counting from 0 to the number of matches
    tracks = np.arange(num_matches)

    # add color column with default value of 1
    df = pd.DataFrame(tracks, columns=["track"])
    df["color"] = 1

    ep.export_ply(df['color'].to_frame(), os.path.join(path_pc_fld, "tracks.ply"))

    # create the path to doc.xml
    xml_path = os.path.join(path_pc_fld, "doc.xml")

    # create the doc.xml
    _create_doc_xml(xml_path, points_per_image, num_matches)

    # zip the folder
    output_zip_path = os.path.join(path_pc_fld, 'point_cloud.zip')
    zp.zip_folder(path_pc_fld, output_zip_path, delete_files=True)

    af.adapt_frame(files_fld, "point_cloud", "path", "point_cloud/point_cloud.zip")

def _generate_ply_files(tp_dict, files_fld):
    """
    Generates PLY files from the tie points dictionary.
    :param tie_points_dict: Dictionary where the key is (image1, image2) and value is a numpy array with shape (x,4).
    """

    num_matches = 0

    image_names = sorted(set([key[0] for key in tp_dict.keys()] + [key[1] for key in tp_dict.keys()]))
    image_to_index = {name: idx for idx, name in enumerate(image_names)}

    # Initialize a dictionary to hold the points for each image
    image_points = {image: [] for image in image_names}
    point_id = 0

    # Iterate over the tie points
    for (image1, image2), points in tp_dict.items():

        if points.shape[0] < 10:
            continue

        num_matches = num_matches + points.shape[0]

        for point in points:
            x1, y1, x2, y2 = point

            # Add point to image1's PLY data
            image_points[image1].append((x1, y1, 10.0, point_id))

            # Add point to image2's PLY data
            image_points[image2].append((x2, y2, 10.0, point_id))

            # Increment the ID for the next point
            point_id += 1

    # Save PLY files for each image
    for image, points in image_points.items():
        index = image_to_index[image]
        file_name = f"p{index}.ply"
        file_path = os.path.join(files_fld, file_name)
        _save_ply_file(file_path, points)
    return num_matches


def _save_ply_file(file_path, points):
    """
    Saves a PLY file with given points.
    :param file_name: Name of the output PLY file.
    :param points: List of points where each point is (x, y, z, id).
    """
    with open(file_path, 'wb') as file:
        # PLY header
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property int id\n"
            "end_header\n"
        )
        file.write(header.encode('ascii'))

        # Write point data
        for point in points:
            file.write(struct.pack('<fffI', point[0], point[1], point[2], point[3]))


def _create_doc_xml(xml_path, points_per_image, num_matches):
    # Create the root element
    root = ET.Element("point_cloud", version="1.2.0")

    # Create the params element
    params = ET.SubElement(root, "params")

    # Add dataType element
    ET.SubElement(params, "dataType").text = "uint8"

    # Add bands element
    bands = ET.SubElement(params, "bands")
    ET.SubElement(bands, "band")

    # Add tracks element
    ET.SubElement(root, "tracks", path="tracks.ply", count=str(num_matches))

    # counter for the camera ply files
    c_counter = 0

    # Add projections elements
    for elem in points_per_image.values():
        # remove duplicates
        _el = np.unique(elem, axis=0)

        ET.SubElement(root, "projections", camera_id=str(c_counter),
                      path=f"p{c_counter}.ply",
                      count=str(_el.shape[0]))

        c_counter += 1

    metadata = {
        "Info/LastSavedDateTime": "2024:08:19 16:30:19",
        "Info/LastSavedSoftwareVersion": "2.1.2.18358",
        "Info/OriginalDateTime": "2024:08:19 16:30:19",
        "Info/OriginalSoftwareVersion": "2.1.2.18358",
        "MatchPhotos/cameras": "",
        "MatchPhotos/descriptor_type": "binary",
        "MatchPhotos/descriptor_version": "1.1.0",
        "MatchPhotos/downscale": "1",
        "MatchPhotos/downscale_3d": "1",
        "MatchPhotos/duration": "2.523574",
        "MatchPhotos/filter_mask": "true",
        "MatchPhotos/filter_stationary_points": "true",
        "MatchPhotos/generic_preselection": "true",
        "MatchPhotos/guided_matching": "false",
        "MatchPhotos/keep_keypoints": "true",
        "MatchPhotos/keypoint_limit": "40000",
        "MatchPhotos/keypoint_limit_3d": "100000",
        "MatchPhotos/keypoint_limit_per_mpx": "1000",
        "MatchPhotos/laser_scans_vertical_axis": "0",
        "MatchPhotos/mask_tiepoints": "true",
        "MatchPhotos/match_laser_scans": "false",
        "MatchPhotos/max_workgroup_size": "100",
        "MatchPhotos/ram_used": "349401088",
        "MatchPhotos/reference_preselection": "true",
        "MatchPhotos/reference_preselection_mode": "0",
        "MatchPhotos/reset_matches": "true",
        "MatchPhotos/subdivide_task": "true",
        "MatchPhotos/tiepoint_limit": "4000",
        "MatchPhotos/workitem_size_cameras": "20",
        "MatchPhotos/workitem_size_pairs": "80",
    }

    # Add the metadata
    meta = ET.SubElement(root, "meta")
    for key, value in metadata.items():
        ET.SubElement(meta, "property", name=key, value=value)

    # Convert the XML structure to a string
    xml_str = ET.tostring(root, encoding='unicode', method='xml')

    # Add XML declaration at the beginning
    xml_declaration = '<?xml version="1.0"?>\n'

    # Combine the XML declaration and the generated XML string
    full_xml = xml_declaration + xml_str

    with open(xml_path, "w") as f:
        f.write(full_xml)