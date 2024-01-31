import os
import shutil

from PIL import Image

import base.load_image_from_file as liff
import base.remove_borders as rb

base_fld = "/data_1/ATM/data_1/"

image_fld = base_fld + "aerial/TMA/downloaded"
image_fld_resampled = base_fld + "aerial/TMA/downloaded_resampled"

mask_fld = base_fld + "aerial/TMA/masked"
img_xml_fld = base_fld + "sfm/xml/images"
cam_xml_fld = base_fld + "sfm/xml/camera"


def create_project_structure(project_folder, input_ids, camera_name,
                             copy_resampled=True,
                             verify=False):

    print("Create Project structure")

    # check if the project folder is really existing
    assert os.path.isdir(project_folder), f"'{project_folder}' is not a path to an existing folder"

    # create folder for homol
    if os.path.isdir(project_folder + "/Homol") is False:
        os.mkdir(project_folder + "/Homol")

    # create folder for the xml files
    if os.path.isdir(project_folder + "/Ori-InterneScan") is False:
        os.mkdir(project_folder + "/Ori-InterneScan")

    # create folder for images resampled
    if os.path.isdir(project_folder + "/images_orig") is False:
        os.mkdir(project_folder + "/images_orig")

    # create folder for images
    if os.path.isdir(project_folder + "/images") is False:
        os.mkdir(project_folder + "/images")

    # create folder for masks
    if os.path.isdir(project_folder + "/masks_orig") is False:
        os.mkdir(project_folder + "/masks_orig")

    # create folder for masks resampled
    if os.path.isdir(project_folder + "/masks") is False:
        os.mkdir(project_folder + "/masks")

    # create folder for the stats
    if os.path.isdir(project_folder + "/stats") is False:
        os.mkdir(project_folder + "/stats")

    # create folder for Ori-Relative
    if os.path.isdir(project_folder + "/Ori-Relative") is False:
        os.mkdir(project_folder + "/Ori-Relative")

    # create folder for Ori-Tapas
    if os.path.isdir(project_folder + "/Ori-Tapas") is False:
        os.mkdir(project_folder + "/Ori-Tapas")

    # copy the image files
    for image_id in input_ids:
        old_path = image_fld + "/" + image_id + ".tif"
        new_path = project_folder + "/" + image_id + ".tif"
        if os.path.isfile(old_path) and os.path.isfile(new_path) is False:
            shutil.copyfile(old_path, new_path)

    # copy (if existing) resampled images
    if copy_resampled:
        for image_id in input_ids:
            old_path = image_fld_resampled + "/OIS-Reech_" + image_id + ".tif"
            new_path = project_folder + "/OIS-Reech_" + image_id + ".tif"
            if os.path.isfile(old_path) and os.path.isfile(new_path) is False:
                shutil.copyfile(old_path, new_path)

                # if copying a resampled file, check if we copied the original
                if os.path.isfile(project_folder + "/" + image_id + ".tif"):
                    shutil.move(project_folder + "/" + image_id + ".tif",
                                project_folder + "/images_orig/" + image_id + ".tif")

    # copy (if existing) masks
    for image_id in input_ids:
        old_path = mask_fld + "/" + image_id + ".tif"
        new_path = project_folder + "/masks_orig/" + image_id + ".tif"
        if os.path.isfile(old_path) and os.path.isfile(new_path) is False:
            shutil.copyfile(old_path, new_path)

    # copy (if existing) xml files
    for image_id in input_ids:
        old_path = img_xml_fld + "/MeasuresIm-" + image_id + ".tif.xml"
        new_path = project_folder + "/Ori-InterneScan/MeasuresIm-" + image_id + ".tif.xml"
        if os.path.isfile(old_path) and os.path.isfile(new_path) is False:
            shutil.copyfile(old_path, new_path)

    # copy (if existing) the camera xml files
    old_lcd_path = cam_xml_fld + "/" + camera_name + "-LocalChantierDescripteur.xml"
    old_mc_path = cam_xml_fld + "/" + camera_name + "-MeasuresCamera.xml"
    new_lcd_path = project_folder + "/MicMac-LocalChantierDescripteur.xml"
    new_mc_path = project_folder + "/Ori-InterneScan/MeasuresCamera.xml"
    if os.path.isfile(old_lcd_path) and os.path.isfile(new_lcd_path) is False:
        shutil.copyfile(old_lcd_path, new_lcd_path)
    if os.path.isfile(old_mc_path) and os.path.isfile(new_mc_path) is False:
        shutil.copyfile(old_mc_path, new_mc_path)

    if verify:

        pass

        # check if all images are existing

        # check if all resampled images are existing

        # check if all images xml files are existing

        # check if all camera xml files are existing

    print("Project structure created")
