"""run the complete process to create a DEM from images with Agisoft"""

# Python imports
import json
import os
import time
import shutil
import sys

# Library imports
import geopandas as gpd
import Metashape
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.ndimage import binary_dilation
from shapely.geometry import Polygon
from tqdm import tqdm

# Local imports
import src.base.check_sky as cs
import src.base.enhance_image as ei
import src.base.find_overlapping_images as foi
import src.base.load_credentials as lc
import src.base.rotate_image as ri
import src.dem.estimate_dem_quality as edq
import src.export.export_thumbnail as eth
import src.export.export_tiff as eti
import src.load.load_image as li
import src.load.load_pointcloud as lp
import src.sfm_agi.other.create_tps_frame as ctf
import src.sfm_agi.snippets.add_gcp_markers as agm
import src.sfm_agi.snippets.add_tp_markers as atm
import src.sfm_agi.snippets.create_adapted_mask as cam
import src.sfm_agi.snippets.create_confidence_array as cca
import src.sfm_agi.snippets.create_difference_dem as cdd
import src.sfm_agi.snippets.create_matching_structure as cms  # noqa: SpellingInspection
import src.sfm_agi.snippets.convert_ply_files as cpf
import src.sfm_agi.snippets.find_gcps as fg
import src.sfm_agi.snippets.find_tie_points_for_sfm as ftp
import src.sfm_agi.snippets.georef_ortho as go
import src.sfm_agi.snippets.save_key_points as skp
import src.sfm_agi.snippets.save_sfm_to_db as sstd  # noqa: SpellingInspection
import src.sfm_agi.snippets.save_tie_points as stp

# ignore some warnings
os.environ['KMP_WARNINGS'] = '0'

# Constants
PATH_PROJECT_FOLDERS = "/data/ATM/data_1/sfm/agi_projects"
DEBUG_MODE = True  # if true errors are raised and not ignored

# Variables for sfm
fixed_focal_length = True
use_rotations_only_for_tps = True
pixel_size = 0.025  # in mm
resolution_relative = 0.001  # in px
resolution_absolute = 2  # in m
matching_method = "overlap"  # which images should be matched (all, sequential, overlap)
min_overlap = 0.5  # the minimum overlap for matching with overlap
custom_matching = True  # if True, custom matching will be used (lightglue)
max_tps = 0
min_tp_confidence = 0.9
tp_type = float
input_mode = "bundler"   # how are custom tiepoints added in metashape, can be 'agi' or 'bundler'
custom_markers = False  # if True, custom markers support the matching of Agisoft
zoom_level_dem = 10  # in m
use_gcp_mask = True  # filter ground control points with a mask
mask_type = "confidence"  # "confidence" or "rock"
min_gcp_confidence = 0.8
mask_buffer = 10  # in pixels

# Other variables
save_text_output = True  # if True, the output will be saved in a text file
save_commands = True  # if True, the arguments of the commands will be saved in a json file
auto_true_for_new = True  # new projects will have all steps set to True
flag_display_steps = True  # an additional flag to enable/disable the display steps
absolute_mode = True  # if False, the execution stops after the relative steps

# Steps
STEPS = {
    "create_masks": False,
    "union_masks": False,
    "enhance_photos": False,
    "match_photos": True,
    "align_cameras": True,
    "build_depth_maps_relative": True,
    "build_pointcloud_relative": True,
    "build_mesh_relative": True,
    "build_dem_relative": True,
    "build_orthomosaic_relative": True,
    "build_confidence_relative": True,
    "georef_output_images": True,
    "create_gcps": True,
    "load_gcps": True,
    "build_depth_maps_absolute": True,
    "build_pointcloud_absolute": True,
    "build_mesh_absolute": True,
    "build_dem_absolute": True,
    "build_orthomosaic_absolute": True,
    "export_alignment": True,
    "build_confidence_absolute": True,
    "build_difference_dem": True,
    "evaluate_dem": True,
    "save_to_psql": False,
}

DEBUG_STEPS = {
    "save_tps_to_csv": True,
    "save_transforms": True,
    "show_connections": True   # add connections lines between the images
}

# Display settings
DISPLAY_STEPS = {
    "save_thumbnails": False,
    "save_key_points": True,
    "save_tie_points": True,
    "save_aoi": True,
}

CACHE_STEPS = {
    'save_masks': True,
    'use_masks': True,
    'save_tps': True,
    'use_tps': True,
}


def run_agi(project_name: str, images: list,
            camera_positions: dict | None = None, camera_accuracies: dict | None = None,
            camera_rotations: dict | None = None, camera_footprints: dict | None = None,
            camera_tie_points: dict | None = None, focal_lengths=None,
            gcp_accuracy: tuple | None = None,
            azimuth: float = None, absolute_bounds: list | None = None,
            epsg_code: int = 3031,
            overwrite: bool = False, resume: bool = False):
    """

    Args:
        project_name (str): The name of the project.
        images (list): A list of image paths.
        camera_positions (dict): A dictionary with camera positions. The keys are the image ids and the values are
            lists with the x, y, z coordinates. Optional.
        camera_accuracies (dict): A dictionary with camera accuracies. The keys are the image ids and the values are
            lists with the x, y, z accuracies. Optional.
        camera_rotations (dict): A dictionary with camera rotations. The keys are the image ids and the values are
            lists with the yaw, pitch, roll rotations. Optional.
        camera_footprints (dict): A dictionary with camera footprints. The keys are the image ids and the values are
            lists with the footprints. Optional.
        camera_tie_points (dict): A dictionary with camera tie points. The keys are the image ids and the values are
            lists with the tie points. Optional.
        focal_lengths (dict): A dictionary with focal lengths. The keys are the image ids and the values are the focal
            lengths. Optional.
        azimuth (list): The azimuth of the images. Optional.
        absolute_bounds (list): The absolute bounds of the project. Order is [min_x, min_y, max_x, max_y]. Optional.
        epsg_code: The EPSG code of the coordinate system. Default is 3031.
        overwrite: If True, the project will be overwritten. Default is False.
        resume: If True, the project will be resumed. Default is False.
    Returns:
        None
    Raises:
        ValueError: If no images are provided.
        ValueError: If both resume and overwrite are set to True.
    """

    # create empty conn variable
    conn = None

    if resume and overwrite:
        raise ValueError("Both RESUME and OVERWRITE cannot be set to True.")

    if custom_markers and custom_matching:
        raise ValueError("Both CUSTOM_MARKERS and CUSTOM_MATCHING cannot be set to True.")

    if matching_method == "overlap" and camera_footprints is None:
        raise ValueError("Camera footprints must be provided if 'overlap' is used as matching method.")

    # set all steps and display steps to True if RESUME is True
    if resume is False and auto_true_for_new:
        print("Auto setting of all steps to True due to RESUME")
        for key in STEPS:
            STEPS[key] = True
        for key in DISPLAY_STEPS:
            DISPLAY_STEPS[key] = True

    # set all display steps to False if flag_display_steps is False
    if flag_display_steps is False:
        print("Auto setting of all display steps to False")
        for key in DISPLAY_STEPS:
            DISPLAY_STEPS[key] = False

    # check if we have image
    if len(images) == 0:
        raise ValueError("No images were provided")

    # check if the number of images is equal to the (optional) dicts
    if focal_lengths is not None and len(images) != len(focal_lengths):
        raise ValueError(f"The number of images ({len(images)}) "
                         f"and focal lengths ({len(focal_lengths)}) do not match.")
    if camera_footprints is not None and len(images) != len(camera_footprints):
        raise ValueError(f"The number of images ({len(images)}) "
                         f"and camera footprints ({len(camera_footprints)}) do not match.")
    if camera_positions is not None and len(images) != len(camera_positions):
        raise ValueError(f"The number of images ({len(images)}) "
                         f"and camera positions ({len(camera_positions)}) do not match.")
    if camera_accuracies is not None and len(images) != len(camera_accuracies):
        raise ValueError(f"The number of images ({len(images)}) "
                         f"and camera accuracies ({len(camera_accuracies)}) do not match.")
    if camera_rotations is not None and len(images) != len(camera_rotations):
        raise ValueError(f"The number of images ({len(images)}) "
                         f"and camera rotations ({len(camera_rotations)}) do not match.")

    # define path to the project folder and the project files
    project_fld = os.path.join(PATH_PROJECT_FOLDERS, project_name)
    project_psx_path = project_fld + "/" + project_name + ".psx"
    project_files_path = project_fld + "/" + project_name + ".files"

    # remove the complete project folder if OVERWRITE is set to True
    if overwrite and os.path.exists(project_fld):
        print(f"Remove '{project_fld}'")
        shutil.rmtree(project_fld)

    # create project folder if not existing
    if os.path.isdir(project_fld) is False:
        os.makedirs(project_fld)

    # create log folder if not existing
    log_fld = os.path.join(project_fld, "log")
    if os.path.isdir(log_fld) is False:
        os.makedirs(log_fld)

    # create argument folder if not existing
    argument_fld = os.path.join(log_fld, "arguments")
    if os.path.isdir(argument_fld) is False:
        os.makedirs(argument_fld)

    if save_text_output:
        # define path to the log file
        path_log = os.path.join(log_fld, "log.txt")
        tee = Tee(path_log)

    try:
        print("STEPS:")
        print(STEPS)

        print("images:")
        print(images)

        # get the license key
        licence_key = lc.load_credentials("agisoft")['licence']

        # Activate the license
        Metashape.License().activate(licence_key)

        # enable use of gpu
        Metashape.app.gpu_mask = 1
        Metashape.app.cpu_enable = False

        # init some path variables
        output_dem_path = None
        output_ortho_path = None

        # create the metashape project
        doc = Metashape.Document(read_only=False)  # noqa

        # check if the project already exists
        if os.path.exists(project_psx_path):

            if resume is False:
                raise FileExistsError("The project already exists. Set RESUME to True to resume the project.")

            # load the project
            doc.open(project_psx_path, ignore_lock=True)

        else:
            # save the project with file path so that later steps can be resumed
            doc.save(project_psx_path)

        # init output folder
        output_fld = os.path.join(project_fld, "output")
        if os.path.isdir(output_fld) is False:
            os.makedirs(output_fld)

        # init data folder
        data_fld = os.path.join(project_fld, "data")
        if os.path.isdir(data_fld) is False:
            os.makedirs(data_fld)

        # init display folder
        if any(DISPLAY_STEPS.values()):  # check if any of the display steps are True
            display_fld = os.path.join(project_fld, "display")
            if os.path.isdir(display_fld) is False:
                os.makedirs(display_fld)
        else:
            display_fld = None

        # init debug folder
        if any(DEBUG_STEPS.values()):  # check if any of the debug steps are True
            debug_fld = os.path.join(project_fld, "debug")
            if os.path.isdir(debug_fld) is False:
                os.makedirs(debug_fld)
        else:
            debug_fld = None

        # init image folder
        img_folder = os.path.join(data_fld, "images")
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        # save which images are rotated
        rotated_images = []

        # check if all images are in already in the image folder
        if len(os.listdir(img_folder)) != len(images):

            tqdm.write("Copy images:")
            time.sleep(0.1)
            start_time = time.time()

            # iterate the images
            for i, image_path in (pbar := tqdm(enumerate(images), total=len(images))):

                # get file name from path
                file_name = os.path.basename(image_path)

                # define output path
                output_path = os.path.join(img_folder, file_name)

                # update the image path
                images[i] = output_path

                # only copy the image if it does not exist
                if os.path.exists(output_path) is False:

                    # copy the image to the output folder
                    shutil.copy(image_path, output_path)

                    # check image rotation and rotate if necessary
                    correct_rotation = cs.check_sky(output_path, conn=conn)

                    if correct_rotation is False:

                        print(file_name, "is not correctly oriented. Rotating image..")

                        # load the image
                        img = li.load_image(output_path)

                        # rotate the image
                        img = ri.rotate_image(img, 180)

                        # save the rotated image
                        eti.export_tiff(img, output_path, overwrite=True)

                        rotated_images.append(file_name[:-4])
            pbar.close()

            finish_time = time.time()
            exec_time = finish_time - start_time
            tqdm.write(f"Copy images - finished ({exec_time:.4f} s)")

        # add a chunk for the doc and add images
        if len(doc.chunks) == 0:
            chunk = doc.addChunk()

            tqdm.write("Add Photos")
            start_time = time.time()

            # add the images to the chunk
            chunk.addPhotos(images, progress=_print_progress)

            finish_time = time.time()
            exec_time = finish_time - start_time
            tqdm.write(f"Add Photos - finished ({exec_time:.4f} s)")

        else:
            chunk = doc.chunks[0]

        if DISPLAY_STEPS["save_thumbnails"]:
            thumb_folder = os.path.join(display_fld, "thumbnails")
            if not os.path.exists(thumb_folder):
                os.makedirs(thumb_folder)

            tqdm.write("Save Thumbnails")
            time.sleep(0.1)
            start_time = time.time()

            for camera in (pbar := tqdm(chunk.cameras)):

                if camera.enabled is False:
                    continue

                thumb_path = os.path.join(thumb_folder, f"{camera.label}_thumb.jpg")

                image = li.load_image(camera.label)
                eth.export_thumbnail(image, thumb_path)
            pbar.close()

            finish_time = time.time()
            exec_time = finish_time - start_time
            tqdm.write(f"Save Thumbnails - finished ({exec_time:.4f} s)")

        # get the image dimensions
        image_dims = {}
        for camera in chunk.cameras:
            if camera.enabled is False:
                continue

            image_dims[camera.label] = [camera.calibration.width, camera.calibration.height]

        # add focal length to the images
        if focal_lengths is not None:

            print("Set Focal length")
            start_time = time.time()

            # set focal length if given
            for camera in chunk.cameras:

                if camera.enabled is False:
                    continue

                # check if focal length is given
                if camera.label in focal_lengths:

                    # get the focal length
                    focal_length = focal_lengths[camera.label]

                    # check validity of focal length
                    if np.isnan(focal_length):
                        print(f"WARNING: Focal length is NaN for {camera.label}")
                        continue

                    print(f" - Set focal length for {camera.label} to {focal_length}")

                    # set the focal length of the camera
                    camera.sensor.focal_length = focal_length
                    camera.sensor.pixel_size = (pixel_size, pixel_size)

                    # set the fixed parameters
                    if fixed_focal_length:
                        camera.sensor.fixed_params = ['f']
                else:
                    print(f"WARNING: Focal length not given for {camera.label}")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Set Focal length - finished ({exec_time:.4f} s)")

            doc.save()

        # add camera positions to the images
        if camera_positions is not None:

            print("Set Camera positions")
            start_time = time.time()

            # if camera positions are given, we need to set the crs
            chunk.crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa

            # iterate over the cameras
            for camera in chunk.cameras:

                if camera.enabled is False:
                    continue

                # check if camera position is given
                if camera.label in camera_positions:

                    # get the camera position values
                    camera_entry = camera_positions[camera.label]
                    x, y, z = camera_entry[0], camera_entry[1], camera_entry[2]

                    print(f" - Set camera position for {camera.label} to ({x}, {y}, {z})")

                    # set the position of the camera
                    camera.reference.location = Metashape.Vector([x, y, z])  # noqa
                else:
                    print(f"WARNING: Camera position not given for {camera.label}")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Set Camera positions - finished ({exec_time:.4f} s)")

            # check for the other dicts
            if camera_accuracies is None:
                print("WARNING: Camera accuracies are not given")
            if camera_rotations is None:
                print("WARNING: Camera rotations are not given")

            doc.save()

        # add camera accuracies to the images
        if camera_accuracies is not None:

            if camera_positions is None:
                raise ValueError("Camera positions must be given if camera accuracies are given.")

            print("Set Camera accuracies")
            start_time = time.time()

            # iterate over the cameras
            for camera in chunk.cameras:

                if camera.enabled is False:
                    continue

                # check if camera accuracy is given
                if camera.label in camera_accuracies:

                    # get the camera position values
                    camera_entry = camera_accuracies[camera.label]
                    acc_x, acc_y, acc_z = camera_entry[0], camera_entry[1], camera_entry[2]

                    print(f" - Set camera accuracy for {camera.label} to ({acc_x}, {acc_y}, {acc_z})")

                    # set the accuracy of the camera
                    camera.reference.accuracy = Metashape.Vector([acc_x, acc_y, acc_z])  # noqa

                else:
                    print(f"WARNING: Camera accuracy not given for {camera.label}")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Set Camera accuracies - finished ({exec_time:.4f} s)")

            doc.save()

        # add camera rotations to the images
        if camera_rotations is not None and use_rotations_only_for_tps is False:

            print("Set Camera rotations")
            start_time = time.time()

            # iterate over the cameras
            for camera in chunk.cameras:

                if camera.enabled is False:
                    continue

                # check if camera rotation is given
                if camera.label in camera_rotations:

                    # get the camera rotation values
                    entry = camera_rotations[camera.label]
                    yaw, pitch, roll = entry[0], entry[1], entry[2]

                    print(f" - Set camera rotation for {camera.label} to ({yaw}, {pitch}, {roll})")

                    # set the rotation of camera if given
                    camera.reference.rotation = Metashape.Vector([yaw, pitch, roll])  # noqa

                else:
                    print(f"WARNING: Camera rotation not given for {camera.label}")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Set Camera rotations - finished ({exec_time:.4f} s)")

            doc.save()

        if STEPS["create_masks"]:

            print("Create masks")
            start_time = time.time()

            # create temporary doc
            doc_temp = Metashape.Document(read_only=False)  # noqa

            # add a chunk for the temporary doc
            chunk_temp = doc_temp.addChunk()

            # add the images to the temporary chunk
            chunk_temp.addPhotos(images, progress=_print_progress)

            # set the images to film cameras
            for camera in chunk_temp.cameras:
                if camera.enabled is False:
                    continue

                camera.sensor.film_camera = True
                camera.sensor.fixed_calibration = True

            arguments = {
                'generate_masks': True,
                'generic_detector': False,
                'frame_detector': True,
                'fiducials_position_corners': False
            }

            # detect fiducials
            chunk_temp.detectFiducials(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("detectFiducials", arguments, argument_fld)

            # calibrate fiducials
            for camera in (pbar := tqdm(chunk_temp.cameras)):
                if camera.enabled is False:
                    continue

                pbar.set_postfix_str(f"Calibrate fiducials for {camera.label}")

                try:
                    camera.sensor.calibrateFiducials(0.025)
                except RuntimeError:
                    print(f"WARNING: Calibration of fiducials failed for {camera.label}")
                    print(f"{camera.label} will be removed from SfM processing.")
                    camera.enabled = False

            # define path to save masks
            mask_folder = os.path.join(data_fld, "masks_original")
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)

            if len(chunk_temp.cameras) != len(chunk.cameras):
                raise ValueError("The number of cameras in the temporary chunk is not equal to the original chunk.")

            # save masks and copy them to the original chunk as well
            for i, camera_temp in (pbar := tqdm(enumerate(chunk_temp.cameras))):

                pbar.set_postfix_str(f"Save mask for {camera_temp.label}")

                if camera_temp.mask is not None:

                    # save the mask
                    mask_path = os.path.join(mask_folder, f"{camera_temp.label}_mask.tif")
                    camera_temp.mask.image().save(mask_path)

                    # copy the mask to the original chunk
                    camera = chunk.cameras[i]
                    camera.mask = camera_temp.mask

            doc.save()

            # remove the temporary doc
            del doc_temp

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Create masks - finished ({exec_time:.4f} s)")

        # remove all cameras that are not enabled
        for camera in chunk.cameras:
            if camera.enabled is False:
                chunk.remove(camera)

        if STEPS["union_masks"]:

            tqdm.write("Union masks")
            time.sleep(0.1)
            start_time = time.time()

            # adapt the path to the mask folder
            mask_folder = os.path.join(data_fld, "masks_adapted")
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)

            # iterate over the cameras
            for camera in (pbar := tqdm(chunk.cameras)):

                if camera.enabled is False:
                    continue

                pbar.set_postfix_str(f"Union mask for {camera.label}")

                # check if camera has a mask
                if camera.mask:

                    # get the mask
                    mask = camera.mask.image()

                    # get the dimensions of the mask
                    m_width = mask.width
                    m_height = mask.height

                    # convert to np array
                    mask_bytes = mask.tostring()
                    existing_mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((m_height, m_width))

                    # get the image id
                    image_id = camera.label.split("_")[0]

                    # check if image_id is in the rotated images
                    if image_id in rotated_images:
                        rotated = True
                    else:
                        rotated = False

                    # create an adapted mask
                    adapted_mask = cam.create_adapted_mask(existing_mask, image_id,
                                                           rotated=rotated, conn=conn)

                    # convert the adapted mask to a metashape image
                    adapted_mask_m = Metashape.Image.fromstring(adapted_mask,
                                                                adapted_mask.shape[1],
                                                                adapted_mask.shape[0],
                                                                channels=' ',
                                                                datatype='U8')

                    # create a mask object
                    mask_obj = Metashape.Mask()
                    mask_obj.setImage(adapted_mask_m)

                    # set the mask to the camera
                    camera.mask = mask_obj

                    # save the adapted mask
                    mask_path = os.path.join(mask_folder, f"{camera.label}_mask.tif")
                    camera.mask.image().save(mask_path)

            pbar.close()

            finish_time = time.time()
            exec_time = finish_time - start_time
            tqdm.write(f"Union masks - finished ({exec_time:.4f} s)")

            # save the project
            doc.save()

        # init variable for enhanced folder
        enhanced_folder = os.path.join(data_fld, "images_enhanced")

        if STEPS["enhance_photos"]:

            tqdm.write("Enhance photos")
            time.sleep(0.1)
            start_time = time.time()

            # create folder for the enhanced images
            if not os.path.exists(enhanced_folder):
                os.makedirs(enhanced_folder)

            # iterate over the cameras
            for camera in (pbar := tqdm(chunk.cameras)):

                if camera.enabled is False:
                    continue

                pbar.set_postfix_str(f"Enhance photo for {camera.label}")

                # get old image path and also set new one
                image_pth = camera.photo.path
                e_image_pth = os.path.join(enhanced_folder, f"{camera.label}.tif")

                # load the image again
                image = li.load_image(image_pth)

                # get the mask if existing
                if camera.mask is not None:
                    m_mask = camera.mask.image()
                    m_height = m_mask.height
                    m_width = m_mask.width
                    mask_bytes = m_mask.tostring()
                    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(
                        (m_height, m_width))
                else:
                    mask = None

                # enhance the image
                enhanced_image = ei.enhance_image(image, mask)

                # save the enhanced image
                eti.export_tiff(enhanced_image, e_image_pth, overwrite=True)

                # update the image in the chunk
                photo = camera.photo.copy()
                photo.path = e_image_pth
                camera.photo = photo
            pbar.close()

            finish_time = time.time()
            exec_time = finish_time - start_time
            tqdm.write(f"Enhance photos - finished ({exec_time:.4f} s)")

            # save the project
            doc.save()

        if DISPLAY_STEPS["save_thumbnails"] and STEPS["enhance_photos"]:

            # create folder for the thumbnails
            thumb_folder = os.path.join(display_fld, "thumbnails")
            if not os.path.exists(thumb_folder):
                os.makedirs(thumb_folder)

            tqdm.write("Save Thumbnails")
            time.sleep(0.1)
            start_time = time.time()

            for camera in (pbar := tqdm(chunk.cameras)):

                if camera.enabled is False:
                    continue

                img_path = os.path.join(data_fld, "images_enhanced", f"{camera.label}.tif")
                thumb_path = os.path.join(thumb_folder, f"{camera.label}_e_thumb.jpg")

                image = li.load_image(img_path)
                eth.export_thumbnail(image, thumb_path)
            pbar.close()

            finish_time = time.time()
            exec_time = finish_time - start_time
            tqdm.write(f"Save Thumbnails - finished ({exec_time:.4f} s)")

        # match photos
        if STEPS["match_photos"]:

            print("Match photos")
            time.sleep(0.1)
            start_time = time.time()

            if custom_matching:

                # check if the mask folder is existing
                mask_folder = os.path.join(data_fld, "masks_adapted")
                if not os.path.exists(mask_folder):
                    mask_folder = None

                # check if there is a folder with enhanced images
                if os.path.exists(enhanced_folder):
                    tp_img_folder = enhanced_folder
                else:
                    tp_img_folder = img_folder

                # find the tie points between images
                tp_dict, conf_dict = ftp.find_tie_points_for_sfm(tp_img_folder,
                                                                 mask_folder=mask_folder,
                                                                 matching_method=matching_method,
                                                                 footprint_dict=camera_footprints,
                                                                 rotation_dict=camera_rotations,
                                                                 tp_type=tp_type,
                                                                 min_overlap=min_overlap,
                                                                 min_conf=min_tp_confidence,
                                                                 max_tps=max_tps,
                                                                 use_cached_tps=CACHE_STEPS['use_tps'],
                                                                 save_tps=CACHE_STEPS['save_tps'])

                # create the ply files for the custom matching
                cms.create_matching_structure(tp_dict, conf_dict,
                                              project_files_path,
                                              img_folder,
                                              input_mode=input_mode)

                # for bundler some extra steps are required
                if input_mode == "bundler":
                    bundler_path = os.path.join(data_fld, "bundler", "bundler.out")
                    chunk.importCameras(bundler_path, format=Metashape.CamerasFormatBundler)
                elif input_mode == "agi":
                    # load the project again to save the custom tie points
                    doc = Metashape.Document(read_only=False)  # noqa
                    doc.open(project_psx_path, ignore_lock=True)
                    chunk = doc.chunks[0]

                import src.sfm_agi.snippets.get_image_stats as gis
                #gis.get_image_stats(chunk)

            else:

                if custom_markers:

                    print("  Find custom markers between images")

                    mask_folder = os.path.join(data_fld, "masks_adapted")
                    if not os.path.exists(mask_folder):
                        mask_folder = None

                    # check if there is a folder with enhanced images
                    if os.path.exists(enhanced_folder):
                        tp_img_folder = enhanced_folder
                    else:
                        tp_img_folder = img_folder

                    # find the tie points between images
                    tp_dict, conf_dict = ftp.find_tie_points_for_sfm(tp_img_folder,
                                                                     mask_folder=mask_folder,
                                                                     matching_method=matching_method,
                                                                     footprint_dict=camera_footprints,
                                                                     min_overlap=min_overlap)

                    # add the tp markers to the chunk
                    atm.add_tp_markers(chunk, tp_dict, conf_dict, reset_markers=True)

                    print("  Find custom markers between images - Finished")

                    doc.save()

                arguments = {
                    'generic_preselection': True,
                    'reference_preselection': True,
                    'keep_keypoints': False,
                    'filter_mask': True,
                    'mask_tiepoints': True,
                    'filter_stationary_points': True,
                    'reset_matches': True
                }

                # create pairs from the cameras
                if matching_method == "all":
                    # nothing must be done here
                    pass
                elif matching_method == "sequential":
                    pairs = []
                    num_cameras = len(chunk.cameras)
                    for i in range(num_cameras - 1):
                        pairs.append((i, i + 1))
                    arguments['pairs'] = pairs
                elif matching_method == "overlap":
                    # get all camera labels as image ids
                    image_ids = [camera.label for camera in chunk.cameras if camera.enabled]

                    # get the footprints as lst
                    footprints_lst = [camera_footprints[image_id] for image_id in image_ids]

                    overlap_dict = foi.find_overlapping_images(image_ids,
                                                               footprints_lst)
                    pairs = []
                    # create a list with all combinations of images
                    for img_id, overlap_lst in overlap_dict.items():

                        # convert id to index
                        img_idx = image_ids.index(img_id)

                        # iterate over all overlapping images
                        for overlap_id in overlap_lst:

                            # skip identical images
                            if img_id == overlap_id:
                                continue

                            # convert id to index
                            overlap_idx = image_ids.index(overlap_id)

                            # check if the combination is already in the list
                            if (overlap_idx, img_idx) in pairs:
                                continue

                            # add the combination to the list
                            pairs.append((img_idx, overlap_idx))

                    arguments['pairs'] = pairs

                else:
                    raise ValueError(f"Matching method {matching_method} not existing")

                # match photos
                chunk.matchPhotos(**arguments)

                # save the arguments of the command
                if save_commands:
                    _save_command_args("matchPhotos", arguments, argument_fld)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Match photos - finished ({exec_time:.4f} s)")

            # save the project
            doc.save()

        if DEBUG_STEPS["save_tps_to_csv"]:

            # get path to point cloud fld (dependent on bundler)
            point_cloud_fld = os.path.join(project_files_path, "0", "0", "point_cloud")
            tracks_path = os.path.join(point_cloud_fld, "point_cloud", "tracks.txt")
            # check if the folder is already unzipped (with tracks.ply)
            if os.path.exists(tracks_path) is False:
                # convert ply files to txt files
                cpf.convert_ply_files(point_cloud_fld, True)

            # create the dataframe with tie_point/tracks
            tps_df = ctf.create_tps_frame(os.path.join(point_cloud_fld, "point_cloud"))

            output_pth = os.path.join(debug_fld, "tie_points.csv")
            tps_df.to_csv(output_pth, index=False)

        # set back to the original images
        if STEPS["enhance_photos"]:

            tqdm.write("Restore original images")
            time.sleep(0.1)
            start_time = time.time()

            # iterate over the cameras
            for camera in (pbar := tqdm(chunk.cameras)):

                if camera.enabled is False:
                    continue

                pbar.set_postfix_str(f"Restore image for {camera.label}")

                # get original image path
                image_pth = os.path.join(img_folder, f"{camera.label}.tif")
                # update the image in the chunk
                photo = camera.photo.copy()
                photo.path = image_pth
                camera.photo = photo
            pbar.close()

            finish_time = time.time()
            exec_time = finish_time - start_time
            tqdm.write(f"Restore original images - finished ({exec_time:.4f} s)")

            # save the project
            doc.save()

        # save key points
        if DISPLAY_STEPS["save_key_points"]:

            print("Save key points")
            start_time = time.time()

            # define point cloud folder
            point_cloud_fld = os.path.join(project_files_path, "0", "0", "point_cloud")

            # define save path
            kp_fld = os.path.join(display_fld, "key_points")
            if not os.path.exists(kp_fld):
                os.makedirs(kp_fld)

            # get image ids
            image_ids = [camera.label for camera in chunk.cameras if camera.enabled]

            # call snippet to save key points
            skp.save_key_points(image_ids, point_cloud_fld, kp_fld)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Save key points - finished ({exec_time:.4f} s)")

        # align cameras
        if STEPS["align_cameras"]:

            print("Align cameras")
            start_time = time.time()

            """
            # check if the cameras are grouped (can be removed with custom matching)
            if len(chunk.camera_groups) == 0:
                print("Create new camera group")
                # create a group with all cameras
                cam_group = chunk.addCameraGroup()
                for camera in chunk.cameras:
                    camera.group = cam_group
                doc.save()
                doc.open(project_psx_path, ignore_lock=True)
            """

            arguments = {
                'reset_alignment': False,
                'adaptive_fitting': True
            }

            if custom_matching and input_mode == "bundler":
                arguments['reset_alignment'] = True

            # align cameras
            chunk.alignCameras(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("alignCameras", arguments, argument_fld)

            # save the project
            doc.save()

            # check how many cameras are aligned
            num_cameras = len(chunk.cameras)
            num_aligned = 0
            for camera in chunk.cameras:
                if camera.transform and camera.enabled:
                    num_aligned += 1

            if num_aligned == 0:
                raise Exception("No cameras are aligned")

            print(f"Aligned cameras: {num_aligned}/{num_cameras}")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Align cameras - finished ({exec_time:.4f} s)")

        # save tie points
        if DISPLAY_STEPS["save_tie_points"]:

            print("Save tie points")
            start_time = time.time()

            tp_fld = os.path.join(display_fld, "tie_points")
            if not os.path.exists(tp_fld):
                os.makedirs(tp_fld)
            stp.save_tie_points(chunk, tp_fld)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Save tie points - finished ({exec_time:.4f} s)")

        # build depth maps
        if STEPS["build_depth_maps_relative"]:

            print("Build relative depth maps")
            start_time = time.time()

            arguments = {}

            chunk.buildDepthMaps(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildDepthMaps_relative", arguments, argument_fld)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative depth maps - finished ({exec_time:.4f} s)")

            # save the project
            doc.save()

        print("Create relative bounding box")
        start_time = time.time()

        # update alignment and transform
        if camera_positions is not None:
            chunk.optimizeCameras()
            chunk.updateTransform()

        # update projection
        if camera_positions is not None:
            projection_absolute = Metashape.OrthoProjection()
            projection_absolute.crs = chunk.crs

        # get center and size of the chunk
        center = chunk.region.center
        size = chunk.region.size

        # Calculate the minimum and maximum corners of the bounding box
        min_corner = Metashape.Vector([center.x - size.x / 2,  # noqa
                                       center.y - size.y / 2,
                                       center.z - size.z / 2])
        max_corner = Metashape.Vector([center.x + size.x / 2,  # noqa
                                       center.y + size.y / 2,
                                       center.z + size.z / 2])

        if camera_positions is not None:
            min_corner = chunk.crs.project(chunk.transform.matrix.mulp(min_corner))
            max_corner = chunk.crs.project(chunk.transform.matrix.mulp(max_corner))

        # restrain the bounding box to the absolute bounds if given
        if camera_positions is not None and absolute_bounds is not None:
            min_corner.x = min(min_corner.x, absolute_bounds[0])
            min_corner.y = min(min_corner.y, absolute_bounds[1])
            max_corner.x = max(max_corner.x, absolute_bounds[2])
            max_corner.y = max(max_corner.y, absolute_bounds[3])

        # create 2d versions of the corners
        min_corner_2d = Metashape.Vector([min_corner.x, min_corner.y])  # noqa
        max_corner_2d = Metashape.Vector([max_corner.x, max_corner.y])  # noqa

        # Create the bounding box
        bounding_box_relative = Metashape.BBox(min_corner_2d, max_corner_2d)  # noqa

        # intermediate check for the bounding box (raise error if all values are 0)
        if (min_corner.x == 0 and min_corner.y == 0 and
                max_corner.x == 0 and max_corner.y == 0):
            raise ValueError("Bounding box is empty. Check the previous steps for completeness.")

        print("Corners of the relative bounding box:")
        print(f"Min: {min_corner}")
        print(f"Max: {max_corner}")

        finish_time = time.time()
        exec_time = finish_time - start_time

        print(f"Create relative bounding box - finished ({exec_time:.4f} s)")

        # build mesh
        if STEPS["build_mesh_relative"]:

            print("Build relative mesh")
            start_time = time.time()

            arguments = {
                'surface_type': Metashape.Arbitrary
            }

            # build mesh
            chunk.buildModel(**arguments)
            doc.save()

            # define output path for the model
            output_model_path = os.path.join(output_fld, project_name + "_model_relative.obj")

            # define export parameters
            arguments = {
                'path': output_model_path,
            }

            # export the model
            chunk.exportModel(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("exportMesh_relative", arguments, argument_fld)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative mesh - finished ({exec_time:.4f} s)")

        # build point cloud
        if STEPS["build_pointcloud_relative"]:

            print("Build relative point cloud")
            start_time = time.time()

            arguments = {
                'point_colors': True,
                'point_confidence': True
            }

            chunk.buildPointCloud(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildPointCloud_relative", arguments, argument_fld)

            # save the project
            doc.save()

            # define output path for the point cloud
            output_pc_path = os.path.join(output_fld, project_name + "_pointcloud_relative.ply")

            arguments = {
                'path': output_pc_path,
            }
            chunk.exportPointCloud(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("exportPointCloud_relative", arguments, argument_fld)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative point cloud - finished ({exec_time:.4f} s)")

        # build DEM
        if STEPS["build_dem_relative"]:

            print("Build relative DEM")
            start_time = time.time()

            # define projection
            projection = Metashape.OrthoProjection()
            projection.crs = chunk.crs

            # arguments for building the DEM
            arguments = {
                'source_data': Metashape.DataSource.PointCloudData,
                'interpolation': Metashape.Interpolation.EnabledInterpolation,
                'projection': projection,
            }

            # add region to build parameters
            if bounding_box_relative is not None:
                arguments['region'] = bounding_box_relative

            # add resolution to build parameters dependent on camera positions
            if camera_positions is None:
                arguments['resolution'] = resolution_relative
            else:
                arguments['resolution'] = resolution_absolute

            # build the DEM
            chunk.buildDem(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildDem_relative", arguments, argument_fld)

            # save the project
            doc.save()

            # define output path for the DEM
            output_dem_path = os.path.join(output_fld, project_name + "_dem_relative.tif")

            # set export parameters for the DEM
            arguments = {
                'path': output_dem_path,
                'source_data': Metashape.ElevationData,
                'image_format': Metashape.ImageFormatTIFF,
                'raster_transform': Metashape.RasterTransformNone,
                'nodata_value': -9999,
            }

            # add region to export parameters
            if bounding_box_relative is not None:
                arguments['region'] = bounding_box_relative

            # add resolution to build parameters dependent on camera positions
            if camera_positions is None:
                arguments['resolution'] = resolution_relative
            else:
                arguments['resolution'] = resolution_absolute

            # export the DEM
            chunk.exportRaster(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("exportDEM_relative", arguments, argument_fld)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative DEM - finished ({exec_time:.4f} s)")

        """
        # Accessing geolocation information for each camera (image)
        for camera in chunk.cameras:
            if camera.reference.location:  # Check if location data is available
                print("Camera:", camera.label)
                print("Location:", camera.reference.location)  # Prints Vector(x, y, z)
                print("Accuracy:", camera.reference.accuracy)  # Prints accuracy if available
            else:
                print("Camera:", camera.label, "has no geolocation data.")
        """

        # build ortho-mosaic
        if STEPS["build_orthomosaic_relative"]:

            print("Build relative orthomosaic")
            start_time = time.time()

            # define projection
            projection = Metashape.OrthoProjection()
            projection.crs = chunk.crs

            # arguments for building the orthomosaic
            arguments = {
                'surface_data': Metashape.ModelData,
                'blending_mode': Metashape.MosaicBlending,
                'projection': projection,
            }

            # add region to build parameters
            if bounding_box_relative is not None:
                arguments['region'] = bounding_box_relative

            # add resolution to build parameters dependent on camera positions
            if camera_positions is None:
                arguments['resolution'] = resolution_relative
            else:
                arguments['resolution'] = resolution_absolute

            # build the orthomosaic
            chunk.buildOrthomosaic(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildOrthomosaic_relative", arguments, argument_fld)

            # save the project
            doc.save()

            # define output path for the orthomosaic
            output_ortho_path = os.path.join(output_fld, project_name + "_ortho_relative.tif")

            # set export parameters for the orthomosaic
            arguments = {
                'path': output_ortho_path,
                'source_data': Metashape.OrthomosaicData,
                'image_format': Metashape.ImageFormatTIFF,
                'raster_transform': Metashape.RasterTransformNone,
                'nodata_value': -9999,
            }

            # add region to export parameters
            if bounding_box_relative is not None:
                arguments['region'] = bounding_box_relative

            # add resolution to build parameters dependent on camera positions
            if camera_positions is None:
                arguments['resolution'] = resolution_relative
            else:
                arguments['resolution'] = resolution_absolute

            # export the orthomosaic
            chunk.exportRaster(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("exportOrthomosaic_relative", arguments, argument_fld)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative orthomosaic - finished ({exec_time:.4f} s)")

        # build confidence array
        if STEPS["build_confidence_relative"]:
            print("Build relative confidence array")
            start_time = time.time()

            # define paths
            output_dem_path = os.path.join(output_fld, project_name + "_dem_relative.tif")
            output_pc_path = os.path.join(output_fld, project_name + "_pointcloud_relative.ply")

            # load the dem & point cloud
            dem, transform = li.load_image(output_dem_path, return_transform=True)
            point_cloud = lp.load_point_cloud(output_pc_path)

            conf_arr = cca.create_confidence_arr(dem, point_cloud, transform,
                                                 interpolate=True, distance=10)

            output_conf_path = os.path.join(output_fld, project_name + "_confidence_relative.tif")

            eti.export_tiff(conf_arr, output_conf_path,
                            transform=transform, overwrite=True)
            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative confidence array - finished ({exec_time:.4f} s)")

        if absolute_mode is False:
            print("Finished processing relative data & absolute mode is disabled")
            return

        if STEPS["georef_output_images"]:

            print("Georeference output images")
            start_time = time.time()

            # use default values for DEM and ortho if not given
            if output_dem_path is None:
                output_dem_path = os.path.join(output_fld, project_name + "_dem_relative.tif")
            if output_ortho_path is None:
                output_ortho_path = os.path.join(output_fld, project_name + "_ortho_relative.tif")

            if os.path.exists(output_dem_path) is False:
                raise FileNotFoundError(f"DEM file does not exist at '{output_dem_path}'")
            if os.path.exists(output_ortho_path) is False:
                raise FileNotFoundError(f"Ortho file does not exist at '{output_ortho_path}'")

            # load the required data
            dem = li.load_image(output_dem_path)
            ortho = li.load_image(output_ortho_path)
            footprints = camera_footprints

            # set nodata to nan
            dem[dem == -9999] = np.nan

            # if ortho has alpha -> remove it
            if len(ortho.shape) == 3:
                ortho = ortho[0, :, :]

            if dem.shape != ortho.shape:
                print("DEM", dem.shape)
                print("Ortho", ortho.shape)
                raise ValueError("DEM and ortho should have the same shape")

            # check which cameras are aligned
            aligned = []
            for camera in chunk.cameras:

                if camera.enabled is False:
                    continue

                if camera.transform:
                    aligned.append(True)
                else:
                    aligned.append(False)

            # create path for the geo-referenced ortho and transform file
            georef_data_fld = os.path.join(data_fld, "georef")
            if os.path.isdir(georef_data_fld) is False:
                os.makedirs(georef_data_fld)
            ortho_georef_path = os.path.join(georef_data_fld, "ortho_georeferenced.tif")
            transform_path = os.path.join(georef_data_fld, "transform.txt")

            # get the transform of dem/ortho (identical for both)
            transform, mask_bounds = go.georef_ortho(ortho, footprints.values(), aligned,
                                                     azimuth=azimuth, auto_rotate=True,
                                                     trim_image=True,
                                                     tp_type=tp_type,
                                                     save_path_ortho=ortho_georef_path,
                                                     save_path_transform=transform_path)

            # try again with complete autorotate
            if transform is None and azimuth is not None:
                transform, mask_bounds = go.georef_ortho(ortho, footprints.values(), aligned,
                                                         azimuth=None, auto_rotate=True,
                                                         trim_image=True,
                                                         tp_type=tp_type,
                                                         save_path_ortho=ortho_georef_path,
                                                         save_path_transform=transform_path)
            if transform is None:
                raise ValueError("Transform is not defined. Check the georef_ortho function.")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Georeference output images - finished ({exec_time:.4f} s)")

        if STEPS["create_gcps"]:

            print("Create GCPs")
            start_time = time.time()

            # define output path in which gcp files are saved
            gcp_fld = os.path.join(data_fld, "georef")
            if not os.path.exists(gcp_fld):
                os.makedirs(gcp_fld)
            gcp_path = os.path.join(gcp_fld, "gcps.csv")

            if bounding_box_relative is None:
                raise ValueError("Bounding box is not defined. Please create a bounding box first.")

            # add resolution to build parameters dependent on camera positions
            if camera_positions is None:
                resolution = resolution_relative
            else:
                resolution = resolution_absolute

            if use_gcp_mask:
                if mask_type == "rock":
                    raise NotImplementedError("Rock mask not implemented yet.")
                    """
                    if use_rock_mask:

                        min_abs_x, min_abs_y = absolute_coords.min(axis=0)
                        max_abs_x, max_abs_y = absolute_coords.max(axis=0)

                        # get the bounds and load the dem
                        absolute_bounds = (min_abs_x, min_abs_y, max_abs_x, max_abs_y)

                        # get the rock mask
                        rock_mask = lrm.load_rock_mask(absolute_bounds, mask_resolution)

                        # give a warning if no rocks are existing in the mask
                        if np.sum(rock_mask) == 0:
                            print("WARNING: No rocks found in the rock mask (find_gcps.py)")

                        # Apply mask buffer by dilating the mask (expanding the regions of 1s)
                        kernel = np.ones((mask_buffer, mask_buffer), dtype=bool)
                        rock_mask = binary_dilation(rock_mask, structure=kernel)

                        y_mask_coords = (df['y_abs'].values - min_abs_y) / mask_resolution
                        x_mask_coords = (df['x_abs'].values - min_abs_x) / mask_resolution

                        # cast to int
                        y_mask_coords = y_mask_coords.astype(int)
                        x_mask_coords = x_mask_coords.astype(int)

                        # Check if the coordinates are within the bounds of the DEM
                        valid_coords = (y_mask_coords >= 0) & (y_mask_coords < rock_mask.shape[0]) & \
                                       (x_mask_coords >= 0) & (x_mask_coords < rock_mask.shape[1])

                        # Filter only the valid points
                        x_mask_coords = x_mask_coords[valid_coords]
                        y_mask_coords = y_mask_coords[valid_coords]
                        df = df[valid_coords]

                        # create the mask
                        mask = rock_mask[y_mask_coords, x_mask_coords] == 1
                    """

                elif mask_type == "confidence":

                    conf_path = os.path.join(output_fld, project_name + "_confidence_relative.tif")
                    conf_arr = li.load_image(conf_path)

                    if conf_arr is None:
                        raise ValueError("Confidence array is not defined. Please create a confidence array first.")

                    gcp_mask = np.zeros_like(dem)
                    gcp_mask[conf_arr > min_gcp_confidence] = 1

                else:
                    raise ValueError(f"Mask type '{mask_type}' not supported.")
            else:
                gcp_mask = None

            # apply buffer to mask
            kernel = np.ones((mask_buffer, mask_buffer), dtype=bool)
            gcp_mask = binary_dilation(gcp_mask, structure=kernel)

            # call snippet to export gcps
            gcp_df = fg.find_gcps(dem, transform,
                                  bounding_box_relative, zoom_level_dem,
                                  resolution, mask=gcp_mask, ortho=ortho)

            # Create labels (n x 1 array)
            num_points = gcp_df.shape[0]
            labels = pd.Series([f"gcp_{i + 1}" for i in range(num_points)])

            gcp_df.insert(0, 'GCP', labels)
            gcp_df.to_csv(gcp_path, sep=';', index=False, float_format='%.8f')

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Create GCPs - finished ({exec_time:.4f} s)")

        if STEPS["load_gcps"]:

            print("Load GCPs")
            start_time = time.time()

            # set path to gpcs
            gcp_path = os.path.join(data_fld, "gcps.csv")

            if os.path.isfile(gcp_path) is False:
                raise FileNotFoundError(f"GCP file does not exist at '{gcp_path}'")

            # load the gcps
            gcps = pd.read_csv(gcp_path, sep=';')

            print(f"{gcps.shape[0]} GCPS are loaded from file")

            # add markers to the chunk
            agm.add_gcp_markers(chunk, gcps, accuracy=gcp_accuracy,
                                epsg_code=epsg_code, reset_markers=True)

            # "https://www.agisoft.com/forum/index.php?topic=7446.0"
            # "https://www.agisoft.com/forum/index.php?topic=10855.0"
            print("FINISHED")
            doc.save()

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Load GCPs - finished ({exec_time:.4f} s)")

        if STEPS["export_alignment"]:

            print("Export alignment")
            start_time = time.time()

            cam_fld = os.path.join(data_fld, "cameras")
            if not os.path.exists(cam_fld):
                os.makedirs(cam_fld)

            # Export camera calibration and orientation
            camera_path = os.path.join(cam_fld, "cameras.txt")
            with open(camera_path, 'w') as f:
                for camera in chunk.cameras:
                    if camera.enabled is False:
                        continue

                    if camera.transform:
                        line = camera.label + ',' + ','.join(map(str, np.asarray(camera.transform).flatten())) + '\n'
                        f.write(line)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Export alignment - finished ({exec_time:.4f} s)")

        print("Set to absolute mode")
        start_time = time.time()

        # set to absolute mode
        chunk.crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa

        # update alignment and transform
        chunk.optimizeCameras()
        chunk.updateTransform()

        projection_absolute = Metashape.OrthoProjection()
        projection_absolute.crs = chunk.crs

        finish_time = time.time()
        exec_time = finish_time - start_time
        print(f"Set to absolute mode - finished ({exec_time:.4f} s)")

        if STEPS["build_depth_maps_absolute"]:

            print("Build absolute depth maps")
            start_time = time.time()

            # arguments for building the depth maps
            arguments = {}

            # build depth maps
            chunk.buildDepthMaps(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildDepthMaps_absolute", arguments, argument_fld)

            # save the project
            doc.save()

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute depth maps - finished ({exec_time:.4f} s)")

        print("Create absolute bounding box")
        start_time = time.time()

        # get center and size of the chunk
        center = chunk.region.center
        size = chunk.region.size

        # Calculate the minimum and maximum corners of the bounding box
        min_corner = Metashape.Vector([center.x - size.x / 2,  # noqa
                                       center.y - size.y / 2,
                                       center.z - size.z / 2])
        max_corner = Metashape.Vector([center.x + size.x / 2,  # noqa
                                       center.y + size.y / 2,
                                       center.z + size.z / 2])

        min_corner_abs = chunk.crs.project(chunk.transform.matrix.mulp(min_corner))
        max_corner_abs = chunk.crs.project(chunk.transform.matrix.mulp(max_corner))

        # restrain the bounding box to the absolute bounds if given
        if absolute_bounds is not None:
            min_corner_abs.x = min(min_corner_abs.x, absolute_bounds[0])
            min_corner_abs.y = min(min_corner_abs.y, absolute_bounds[1])
            max_corner_abs.x = max(max_corner_abs.x, absolute_bounds[2])
            max_corner_abs.y = max(max_corner_abs.y, absolute_bounds[3])

        # create 2d vectors
        min_corner_2d = Metashape.Vector([min_corner_abs.x, min_corner_abs.y])  # noqa
        max_corner_2d = Metashape.Vector([max_corner_abs.x, max_corner_abs.y])  # noqa

        # Create the bounding box
        bounding_box_absolute = Metashape.BBox(min_corner_2d, max_corner_2d)  # noqa

        print("Corners of the absolute bounding box:")
        print(f"Min: {min_corner_2d}")
        print(f"Max: {max_corner_2d}")

        finish_time = time.time()
        exec_time = finish_time - start_time
        print(f"Create absolute bounding box - finished ({exec_time:.4f} s)")

        # build mesh
        if STEPS["build_mesh_absolute"]:

            print("Build absolute mesh")
            start_time = time.time()

            # arguments for building the mesh
            arguments = {
                'surface_type': Metashape.Arbitrary,
                'replace_asset': True
            }

            # build mesh
            chunk.buildModel(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildModel_absolute", arguments, argument_fld)

            # save the project
            doc.save()

            # define output path for the model
            output_model_path = os.path.join(output_fld, project_name + "_model_absolute.obj")

            # define export parameters
            arguments = {
                'path': output_model_path,
            }

            # export the model
            chunk.exportModel(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("exportModel_absolute", arguments, argument_fld)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute mesh - finished ({exec_time:.4f} s)")

        if DISPLAY_STEPS["save_aoi"]:
            print("Save AOI")
            start_time = time.time()

            aoi_folder = os.path.join(data_fld, "aoi")
            if not os.path.exists(aoi_folder):
                os.makedirs(aoi_folder)

            # define path to save the aoi
            aoi_path = os.path.join(aoi_folder, "aoi.shp")

            min_x = min_corner_abs.x  # noqa
            min_y = min_corner_abs.y
            max_x = max_corner_abs.x  # noqa
            max_y = max_corner_abs.y

            # create shapely polygon from bounding box
            aoi = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y), (min_x, min_y)])

            # Define the CRS using the EPSG code
            crs = CRS.from_epsg(epsg_code)

            # Create a GeoDataFrame with the polygon and CRS
            gdf = gpd.GeoDataFrame(index=[0], crs=crs.to_wkt(), geometry=[aoi])

            # Save the GeoDataFrame as a shapefile
            gdf.to_file(aoi_path)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Save AOI - finished ({exec_time:.4f} s)")

        if STEPS["build_pointcloud_absolute"]:

            print("Build absolute point cloud")
            start_time = time.time()

            # build parameters for the point cloud
            arguments = {
                'point_colors': True,
                'point_confidence': True,
                'replace_asset': True
            }

            # build dense cloud
            chunk.buildPointCloud(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildPointCloud_absolute", arguments, argument_fld)

            # save the project
            doc.save()

            # define output path for the point cloud
            output_pc_path = os.path.join(output_fld, project_name + "_pointcloud_absolute.ply")

            # export parameters for the point cloud
            arguments = {
                'path': output_pc_path,
                'crs': chunk.crs,
                'save_point_color': True,
                'save_point_confidence': True
            }

            # export the point cloud
            chunk.exportPointCloud(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("exportPointCloud_absolute", arguments, argument_fld)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute point cloud - finished ({exec_time:.4f} s)")

        # build DEM
        if STEPS["build_dem_absolute"]:

            print("Build absolute DEM")
            start_time = time.time()

            # define projection
            projection = Metashape.OrthoProjection()
            projection.crs = chunk.crs

            # set build parameters for the DEM
            arguments = {
                'source_data': Metashape.DataSource.PointCloudData,
                'interpolation': Metashape.Interpolation.EnabledInterpolation,
                'projection': projection,
                'resolution': resolution_absolute,
                'replace_asset': True
            }

            # add region to build parameters
            if bounding_box_absolute is not None:
                arguments['region'] = bounding_box_absolute

            # build the DEM
            chunk.buildDem(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildDem_absolute", arguments, argument_fld)

            # save the project
            doc.save()

            # define output path for the DEM
            output_dem_path = os.path.join(output_fld, project_name + "_dem_absolute.tif")

            # set export parameters for the DEM
            arguments = {
                'path': output_dem_path,
                'source_data': Metashape.ElevationData,
                'image_format': Metashape.ImageFormatTIFF,
                'raster_transform': Metashape.RasterTransformNone,
                'nodata_value': -9999,
                'projection': projection_absolute,
                'resolution': resolution_absolute
            }

            # add region to export parameters
            if bounding_box_absolute is not None:
                arguments['region'] = bounding_box_absolute

            # export the DEM
            chunk.exportRaster(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("exportDEM_absolute", arguments, argument_fld)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute DEM - finished ({exec_time:.4f} s)")

        # build ortho-mosaic
        if STEPS["build_orthomosaic_absolute"]:

            print("Build absolute orthomosaic")
            start_time = time.time()

            # arguments for building the orthomosaic
            arguments = {
                'surface_data': Metashape.ModelData,
                'blending_mode': Metashape.MosaicBlending,
                'projection': projection_absolute,
                'resolution': resolution_absolute,
                'replace_asset': True
            }

            # add region to build parameters
            if bounding_box_absolute is not None:
                arguments['region'] = bounding_box_absolute

            # build the orthomosaic
            chunk.buildOrthomosaic(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildOrthomosaic_absolute", arguments, argument_fld)

            # save the project
            doc.save()

            # define output path for the orthomosaic
            output_ortho_path = os.path.join(output_fld, project_name + "_ortho_absolute.tif")

            # set export parameters for the orthomosaic
            arguments = {
                'path': output_ortho_path,
                'source_data': Metashape.OrthomosaicData,
                'image_format': Metashape.ImageFormatTIFF,
                'raster_transform': Metashape.RasterTransformNone,
                'nodata_value': -9999,
                'resolution': resolution_absolute
            }

            # add region to export parameters
            if bounding_box_absolute is not None:
                arguments['region'] = bounding_box_absolute

            # export the orthomosaic
            chunk.exportRaster(**arguments)

            if save_commands:
                _save_command_args("exportOrthomosaic_absolute", arguments, argument_fld)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute orthomosaic - finished ({exec_time:.4f} s)")

        # build confidence array
        if STEPS["build_confidence_absolute"]:
            print("Build absolute confidence array")
            start_time = time.time()

            # define paths
            output_dem_path = os.path.join(output_fld, project_name + "_dem_absolute.tif")
            output_pc_path = os.path.join(output_fld, project_name + "_pointcloud_absolute.ply")

            # load the dem & point cloud
            dem, transform = li.load_image(output_dem_path, return_transform=True)
            point_cloud = lp.load_point_cloud(output_pc_path)

            conf_arr = cca.create_confidence_arr(dem, point_cloud, transform,
                                                 interpolate=True, distance=10)

            output_conf_path = os.path.join(output_fld, project_name + "_confidence_absolute.tif")

            eti.export_tiff(conf_arr, output_conf_path, transform=transform, overwrite=True)
            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute confidence array - finished ({exec_time:.4f} s)")

        # init empty quality dict
        quality_dict = None

        if STEPS["build_difference_dem"]:

            # load the historic dem
            historic_dem_path = os.path.join(output_fld, project_name + "_dem_absolute.tif")
            historic_dem, transform = li.load_image(historic_dem_path, return_transform=True)

            min_x = min_corner_abs.x  # noqa
            min_y = min_corner_abs.y
            max_x = max_corner_abs.x  # noqa
            max_y = max_corner_abs.y
            historic_bounds = [min_x, min_y, max_x, max_y]

            # modern dem will be loaded in the function itself
            modern_dem = None

            # get relative dem and define the output path
            difference_dem_rel = cdd.create_difference_dem(historic_dem,
                                                           modern_dem,
                                                           historic_bounds,
                                                           "REMA10")
            relative_path = os.path.join(output_fld,
                                         project_name + "_diff_rel.tif")

            print(np.amin(difference_dem_rel), np.amax(difference_dem_rel))

            # export the relative dem
            eti.export_tiff(difference_dem_rel, relative_path,
                           transform=transform, overwrite=True, no_data=np.nan)

            # make relative dem absolute and define the output path
            difference_dem_abs = np.abs(difference_dem_rel)
            absolute_path = os.path.join(output_fld,
                                            project_name + "_diff_abs.tif")

            print(np.amin(difference_dem_abs), np.amax(difference_dem_abs))

            # export the absolute dem
            eti.export_tiff(difference_dem_abs, absolute_path,
                            transform=transform, overwrite=True,
                            no_data=np.nan)

        if STEPS["evaluate_dem"]:
            print("Evaluate dem")
            start_time = time.time()

            # load the historic dem
            historic_dem_path = os.path.join(output_fld, project_name + "_dem_absolute.tif")
            historic_dem = li.load_image(historic_dem_path)

            # create historic bounds
            min_x = min_corner_abs.x  # noqa
            min_y = min_corner_abs.y
            max_x = max_corner_abs.x  # noqa
            max_y = max_corner_abs.y
            historic_bounds = [min_x, min_y, max_x, max_y]

            # load the confidence dem
            confidence_dem_path = os.path.join(output_fld, project_name + "_confidence_absolute.tif")
            conf_arr = li.load_image(confidence_dem_path)

            # load the modern dem
            modern_dem = None  # will be loaded in the function itself

            # estimate the quality of the DEM
            quality_dict = edq.estimate_dem_quality(historic_dem, modern_dem, conf_arr,
                                                    historic_bounds=historic_bounds,
                                                    modern_source="REMA10")

            # export the quality dict to a json file
            quality_path = os.path.join(output_fld, project_name + "_quality.json")
            with open(quality_path, 'w') as f:
                json.dump(quality_dict, f, indent=4)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Evaluate dem - finished ({exec_time:.4f} s)")

        if STEPS["save_to_psql"]:

            if quality_dict is None:
                raise ValueError("Quality dict must be created before saving to psql.")

            sstd.save_sfm_to_db(project_name, quality_dict, images)

    except Exception as e:
        print("!!ERROR!!")
        print(e)
        if DEBUG_MODE:
            raise e
    finally:
        if save_text_output:
            tee.close()  # noqa


# logging class
class Tee:
    """Class to redirect stdout to a log file"""
    def __init__(self, filename, mode='a'):
        self.terminal = sys.stdout
        self.log = open(filename, mode)
        sys.stdout = self
        sys.stdout = self

    def write(self, message):
        """Write message to both screen and log"""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # ensure real-time logging

    def flush(self):
        """Flush the internal buffer"""
        # needed for internal buffer flushes of python
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """Close the log file"""
        # close the log file when done
        self.log.close()
        sys.stdout = self.terminal
        sys.stderr = self.terminal


def _print_progress(val):
    print('Current task progress: {:.2f}%'.format(val))


def _save_command_args(method, args, output_fld):
    # some arguments are not serializable and must be changed
    if 'blending_mode' in args:
        args['blending_mode'] = str(args['blending_mode']).split('.')[-1]
    if 'crs' in args:
        args['crs'] = str(args['crs']).split(':')[-1][:-3]
    if 'image_format' in args:
        args['image_format'] = str(args['image_format']).split('.')[-1]
    if 'interpolation' in args:
        args['interpolation'] = str(args['interpolation']).split('.')[-1]
    if 'projection' in args:
        projection = str(args['projection'])
        projection = projection.replace('>', '').replace('<', '').replace(' ', '_').replace("'", "")
        args['projection'] = projection
    if 'raster_transform' in args:
        args['raster_transform'] = str(args['raster_transform']).split('.')[-1]
    if 'region' in args:
        bbox = args['region']
        bbox_str = f"{bbox.min[0]},{bbox.min[1]},{bbox.max[0]},{bbox.max[1]}"
        args['region'] = bbox_str
    if 'source_data' in args:
        args['source_data'] = str(args['source_data']).split('.')[-1]
    if 'surface_data' in args:
        args['surface_data'] = str(args['surface_data']).split('.')[-1]
    if 'surface_type' in args:
        args['surface_type'] = str(args['surface_type']).split('.')[-1]

    if len(args) > 0:
        print("Saved commands for", method)
        print(args)
    else:
        print("No commands to save for", method)

    # define path to save the command arguments
    output_path = os.path.join(output_fld, f"{method}_args.json")

    # save the dict to a json file
    with open(output_path, 'w') as f:
        json.dump(args, f, indent=4)


if __name__ == "__main__":
    sys_project_name = sys.argv[1]
    sys_images = json.loads(sys.argv[2])
    sys_camera_positions = json.loads(sys.argv[3])
    sys_camera_accuracies = json.loads(sys.argv[4])
    sys_camera_rotations = json.loads(sys.argv[5])
    sys_camera_footprints = json.loads(sys.argv[6])
    sys_camera_tie_points = json.loads(sys.argv[7])
    sys_focal_lengths = json.loads(sys.argv[8])
    sys_azimuth = json.loads(sys.argv[9])
    sys_absolute_bounds = json.loads(sys.argv[10])
    sys_overwrite = json.loads(sys.argv[11])
    sys_resume = json.loads(sys.argv[12])

    run_agi(sys_project_name, sys_images, sys_camera_positions,
            sys_camera_accuracies, sys_camera_rotations,
            sys_camera_footprints, sys_camera_tie_points,
            sys_focal_lengths, sys_azimuth, sys_absolute_bounds,
            sys_overwrite, sys_resume)
