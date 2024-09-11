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
from shapely.geometry import Polygon
from tqdm import tqdm

# Local imports
import src.base.check_sky as cs
import src.base.enhance_image as ei
import src.base.load_credentials as lc
import src.base.rotate_image as ri
import src.dem.estimate_dem_quality as edq
import src.export.export_thumbnail as eth
import src.export.export_tiff as eti
import src.load.load_image as li
import src.load.load_pointcloud as lp
import src.sfm_agi.snippets.create_adapted_mask as cam
import src.sfm_agi.snippets.create_confidence_dem as ccd
import src.sfm_agi.snippets.create_custom_tie_points as cctp  # noqa
import src.sfm_agi.snippets.export_gcps as fg
import src.sfm_agi.snippets.georef_ortho as go
import src.sfm_agi.snippets.save_key_points as skp
import src.sfm_agi.snippets.save_sfm_to_db as sstd  # noqa
import src.sfm_agi.snippets.save_tie_points as stp

# ignore some warnings
os.environ['KMP_WARNINGS'] = '0'

# Constants
PATH_PROJECT_FOLDERS = "/data/ATM/data_1/sfm/agi_projects"
DEBUG_MODE = True  # if true errors are raised

# Variables for sfm
fixed_focal_length = True
resolution_relative = 0.001  # in px
resolution_absolute = 2  # in m
custom_matching = False  # if True, custom matching will be used (lightglue)
zoom_level_dem = 10  # in m

# Other variables
save_text_output = True  # if True, the output will be saved in a text file
save_commands = True  # if True, the arguments of the commands will be saved in a json file
auto_true_for_new = True  # new projects will have all steps set to True
use_rock_mask = True

# Steps
STEPS = {
    "create_masks": False,
    "union_masks": False,
    "enhance_photos": False,
    "match_photos": False,
    "align_cameras": False,
    "build_depth_maps_relative": False,
    "build_pointcloud_relative": False,
    "build_mesh_relative": False,
    "build_dem_relative": False,
    "build_orthomosaic_relative": False,
    "create_gcps": True,
    "load_gcps": True,
    "build_depth_maps_absolute": True,
    "build_pointcloud_absolute": True,
    "build_mesh_absolute": True,
    "build_dem_absolute": True,
    "build_orthomosaic_absolute": True,
    "export_alignment": True,
    "build_confidence_dem": True,
    "evaluate_dem": True,
}

# Display settings
DISPLAY_STEPS = {
    "save_thumbnails": False,
    "save_masks": False,
    "save_adapted_masks": False,
    "save_key_points": False,
    "save_tie_points": False,
    "save_aoi": False,
}


def run_agi(project_name: str, images: list,
            camera_positions: dict | None = None, camera_accuracies: dict | None = None,
            camera_rotations: dict | None = None, camera_footprints: dict | None = None,
            camera_tie_points: dict | None = None, focal_lengths=None,
            azimuth: float = None, absolute_bounds=None,
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

    # set all steps and display steps to True if RESUME is True
    if resume is False and auto_true_for_new:
        print("Auto setting of all steps to True due to RESUME")
        for key in STEPS:
            STEPS[key] = True
        for key in DISPLAY_STEPS:
            DISPLAY_STEPS[key] = True

    # check if we have image
    if len(images) == 0:
        raise ValueError("No images were provided")

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
        display_fld = os.path.join(project_fld, "display")
        if os.path.isdir(display_fld) is False:
            os.makedirs(display_fld)

        # init image folder
        img_folder = os.path.join(data_fld, "images")
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        # save which images are rotated
        rotated_images = []

        # check if all images are in already in the image folder
        if len(os.listdir(img_folder)) != len(images):

            print("Copy images:")
            start_time = time.time()

            # iterate the images
            for i, image_path in tqdm(enumerate(images), total=len(images)):

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

                    # load the image
                    img = li.load_image(output_path)

                    # rotate the image
                    img = ri.rotate_image(img, 180)

                    # save the rotated image
                    eti.export_tiff(img, output_path, overwrite=True)

                    if correct_rotation is False:
                        rotated_images.append(file_name[:-4])

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Copy images - finished ({exec_time:.4f} s)")

        # add a chunk for the doc and add images
        if len(doc.chunks) == 0:
            chunk = doc.addChunk()

            print("Add Photos")
            start_time = time.time()

            # add the images to the chunk
            chunk.addPhotos(images, progress=_print_progress)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Add Photos - finished ({exec_time:.4f} s)")

        else:
            chunk = doc.chunks[0]

        if DISPLAY_STEPS["save_thumbnails"]:
            thumb_folder = os.path.join(display_fld, "thumbnails")
            if not os.path.exists(thumb_folder):
                os.makedirs(thumb_folder)

            print("Save Thumbnails")
            start_time = time.time()

            for camera in tqdm(chunk.cameras):
                thumb_path = os.path.join(thumb_folder, f"{camera.label}_thumb.jpg")

                image = li.load_image(camera.label)
                eth.export_thumbnail(image, thumb_path)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Save Thumbnails - finished ({exec_time:.4f} s)")

        # add focal length to the images
        if focal_lengths is not None:

            print("Set Focal length")
            start_time = time.time()

            # set focal length if given
            for camera in chunk.cameras:

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

        if camera_rotations is not None:

            print("Set Camera rotations")
            start_time = time.time()

            # iterate over the cameras
            for camera in chunk.cameras:

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
            for camera in chunk_temp.cameras:
                camera.sensor.calibrateFiducials(0.025)

            if DISPLAY_STEPS["save_masks"]:

                # define path to save masks
                mask_folder = os.path.join(data_fld, "masks_original")
                if not os.path.exists(mask_folder):
                    os.makedirs(mask_folder)

                # save masks and copy them to the original chunk as well
                for i, camera_temp in enumerate(chunk_temp.cameras):
                    if camera_temp.mask is not None:
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

        if STEPS["union_masks"]:

            print("Union masks")
            start_time = time.time()

            # iterate over the cameras
            for camera in (pbar := tqdm(chunk.cameras)):

                pbar.set_postfix_str(f"Union mask for {camera.label}")

                # check if camera has a mask
                if camera.enabled and camera.mask:

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

            # save the project
            doc.save()

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Union masks - finished ({exec_time:.4f} s)")

        # init variable for enhanced folder
        enhanced_folder = None

        if STEPS["enhance_photos"]:

            print("Enhance photos")
            start_time = time.time()

            # create folder for the enhanced images
            enhanced_folder = os.path.join(data_fld, "images_enhanced")
            if not os.path.exists(enhanced_folder):
                os.makedirs(enhanced_folder)

            # iterate over the cameras
            for camera in (pbar := tqdm(chunk.cameras)):
                pbar.set_postfix_str(f"Enhance photo for {camera.label}")

                # get old image path and also set new one
                image_pth = camera.photo.path
                e_image_pth = os.path.join(enhanced_folder, f"{camera.label}.tif")

                # load the image again
                image = li.load_image(image_pth)

                # get the mask
                m_mask = camera.mask.image()
                m_height = m_mask.height
                m_width = m_mask.width
                mask_bytes = m_mask.tostring()
                mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(
                    (m_height, m_width))

                # enhance the image
                enhanced_image = ei.enhance_image(image, mask)

                # save the enhanced image
                eti.export_tiff(enhanced_image, e_image_pth, overwrite=True)

                # update the image in the chunk
                photo = camera.photo.copy()
                photo.path = e_image_pth
                camera.photo = photo

            # save the project
            doc.save()

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Enhance photos - finished ({exec_time:.4f} s)")

        if DISPLAY_STEPS["save_thumbnails"] and STEPS["enhance_photos"]:
            thumb_folder = os.path.join(display_fld, "thumbnails")
            if not os.path.exists(thumb_folder):
                os.makedirs(thumb_folder)

            print("Save Thumbnails")
            start_time = time.time()

            for camera in tqdm(chunk.cameras):
                thumb_path = os.path.join(thumb_folder, f"{camera.label}_e_thumb.jpg")

                image = li.load_image(camera.label)
                eth.export_thumbnail(image, thumb_path)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Save Thumbnails - finished ({exec_time:.4f} s)")

        # save masks
        if DISPLAY_STEPS["save_adapted_masks"]:

            print("Save adapted masks")
            start_time = time.time()

            mask_folder = os.path.join(data_fld, "masks_adapted")
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)

            for camera in (pbar := tqdm(chunk.cameras)):

                pbar.set_postfix_str(f"Save mask for {camera.label}")

                if camera.mask is not None:
                    mask_path = os.path.join(mask_folder, f"{camera.label}_mask_adapted.tif")
                    camera.mask.image().save(mask_path)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Save adapted masks - finished ({exec_time:.4f} s)")

        # match photos
        if STEPS["match_photos"]:

            print("Match photos")
            start_time = time.time()

            if custom_matching:

                mask_folder = os.path.join(data_fld, "masks_adapted")
                if not os.path.exists(mask_folder):
                    mask_folder = None

                if STEPS["enhance_photos"]:
                    tp_img_folder = enhanced_folder
                else:
                    tp_img_folder = img_folder

                cctp.create_custom_tie_points(chunk, project_files_path,
                                              tp_img_folder, mask_folder=mask_folder)

                # load the project again to save the custom tie points
                doc.open(project_psx_path, ignore_lock=True)

            else:

                # create pairs from the cameras
                pairs = []
                num_cameras = len(chunk.cameras)
                for i in range(num_cameras - 1):
                    pairs.append((i, i + 1))

                arguments = {
                    'generic_preselection': True,
                    'reference_preselection': True,
                    'keep_keypoints': True,
                    'pairs': pairs,
                    'filter_mask': True,
                    'mask_tiepoints': True,
                    'filter_stationary_points': True,
                    'reset_matches': True
                }

                # match photos
                chunk.matchPhotos(**arguments)

                # save the arguments of the command
                if save_commands:
                    _save_command_args("matchPhotos", arguments, argument_fld)

            # save the project
            doc.save()

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Match photos - finished ({exec_time:.4f} s)")

        # set back to the original images
        if STEPS["enhance_photos"]:

            print("Restore original images")
            start_time = time.time()

            # iterate over the cameras
            for camera in (pbar := tqdm(chunk.cameras)):
                pbar.set_postfix_str(f"Restore image for {camera.label}")

                # get original image path
                image_pth = os.path.join(img_folder, f"{camera.label}.tif")
                # update the image in the chunk
                photo = camera.photo.copy()
                photo.path = image_pth
                camera.photo = photo

            # save the project
            doc.save()

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Restore original images - finished ({exec_time:.4f} s)")

        for camera in chunk.cameras:
            print(camera.label)
            print(camera.photo.path)

        # save key points
        if DISPLAY_STEPS["save_key_points"]:

            print("Save key points")
            start_time = time.time()

            # define save path
            kp_fld = os.path.join(display_fld, "key_points")
            if not os.path.exists(kp_fld):
                os.makedirs(kp_fld)

            # get image ids
            image_ids = [camera.label for camera in chunk.cameras]

            # call snippet to save key points
            skp.save_key_points(image_ids, project_fld, kp_fld)

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

            print(chunk.camera_groups)

            arguments = {
                'reset_alignment': False,
                'adaptive_fitting': True
            }

            # align cameras
            chunk.alignCameras(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("alignCameras", arguments, argument_fld)

            # save the project
            doc.save()

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

            # save the project
            doc.save()

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative depth maps - finished ({exec_time:.4f} s)")

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

            print(arguments['region'])

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

        if STEPS["create_gcps"]:

            print("Create GCPs")
            start_time = time.time()

            # use default values for DEM and ortho if not given
            if output_dem_path is None:
                output_dem_path = os.path.join(output_fld, project_name + "_dem_relative.tif")
            if output_ortho_path is None:
                output_ortho_path = os.path.join(output_fld, project_name + "_ortho_relative.tif")

            # load the required data
            dem = li.load_image(output_dem_path)
            ortho = li.load_image(output_ortho_path)
            footprints = camera_footprints

            if dem.shape != ortho.shape[1:]:
                print("DEM", dem.shape)
                print("Ortho", ortho.shape)
                raise ValueError("DEM and ortho should have the same shape")

            # check which cameras are aligned
            aligned = []
            for camera in chunk.cameras:
                if camera.transform:
                    aligned.append(True)
                else:
                    aligned.append(False)

            # get the transform of dem/ortho (identical for both)
            transform = go.georef_ortho(ortho, footprints.values(), aligned,
                                        azimuth=azimuth, auto_rotate=True)

            # define output path in which gcp files are saved
            gcp_path = os.path.join(data_fld, "gcps.csv")

            if bounding_box_relative is None:
                raise ValueError("Bounding box is not defined. Please create a bounding box first.")

            # add resolution to build parameters dependent on camera positions
            if camera_positions is None:
                resolution = resolution_relative
            else:
                resolution = resolution_absolute

            # call snippet to export gcps
            gcp_df = fg.find_gcps(dem, transform,
                                  bounding_box_relative, zoom_level_dem,
                                  resolution, use_rock_mask=use_rock_mask)

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

            # load gcps from file
            gcp_path = os.path.join(data_fld, "gcps.csv")
            gcps = pd.read_csv(gcp_path, sep=';')

            # set crs of markers
            chunk.marker_crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa
            chunk.marker_location_accuracy = [25, 25, 100]

            # multiply the number of gcps by the number of cameras to get the total number of possible markers
            ttl = gcps.shape[0] * len(chunk.cameras)
            pbar = tqdm(total=ttl)

            # remove all existing markers
            chunk.markers.clear()

            # iterate over the gcp dataframe
            for _, row in gcps.iterrows():

                # don't add gcps that are likely wrong
                if row['z_abs'] < -50:
                    continue

                # create 3d point
                point_3d = Metashape.Vector([row['x_rel'], row['y_rel'], row['z_rel']])  # noqa

                # transform the point to local coordinates
                point_local = chunk.transform.matrix.inv().mulp(point_3d)

                marker = None

                # iterate over the cameras
                for camera in chunk.cameras:

                    # update progress bar
                    pbar.update(1)

                    # check if camera is aligned
                    if not camera.transform:
                        continue

                    # project the point to the camera
                    projection = camera.project(point_local)

                    # check if we have a valid projection
                    if projection is None:
                        continue

                    # get the x and y coordinates
                    x, y = projection

                    # check if the point is within the image
                    if ((0 <= x <= camera.image().width) and
                            (0 <= y <= camera.image().height)):

                        # marker must be created only once
                        if marker is None:
                            marker = chunk.addMarker()
                            marker.label = row['GCP']

                            # set the reference location for the marker
                            marker.reference.location = Metashape.Vector(
                                [row['x_abs'], row['y_abs'], row['z_abs']])  # noqa

                        # set relative projection for the marker
                        m_proj = Metashape.Marker.Projection(Metashape.Vector([x, y]), True)  # noqa
                        marker.projections[camera] = m_proj  # noqa

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

            # Export camera calibration and orientation
            camera_path = os.path.join(data_fld, "cameras.txt")
            with open(camera_path, 'w') as f:
                for camera in chunk.cameras:
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

            # define path to save the aoi
            aoi_path = os.path.join(data_fld, "aoi.shp")

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

        # build confidence
        if STEPS["build_confidence_dem"]:
            print("Build confidence dem")
            start_time = time.time()

            # define paths
            output_dem_path = os.path.join(output_fld, project_name + "_dem_absolute.tif")
            output_pc_path = os.path.join(output_fld, project_name + "_pointcloud_absolute.ply")

            # load the dem & point cloud
            dem, transform = li.load_image(output_dem_path, return_transform=True)
            point_cloud = lp.load_point_cloud(output_pc_path)

            conf_dem = ccd.create_confidence_dem(dem, point_cloud, transform,
                                                 interpolate=True, distance=10)

            output_conf_path = os.path.join(output_fld, project_name + "_confidence_dem.tif")

            eti.export_tiff(conf_dem, output_conf_path, transform=transform)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build confidence dem - finished ({exec_time:.4f} s)")

        # init empty quality dict
        quality_dict = None

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
            confidence_dem_path = os.path.join(output_fld, project_name + "_confidence_dem.tif")
            conf_dem = li.load_image(confidence_dem_path)

            # load the modern dem
            modern_dem = None  # will be loaded in the function itself

            quality_dict = edq.estimate_dem_quality(historic_dem, modern_dem, conf_dem,
                                                    historic_bounds=historic_bounds,
                                                    modern_source="REMA10")

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
