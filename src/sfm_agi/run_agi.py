"""run the complete process to create a DEM from images with Agisoft"""

# Python imports
import cv2
import json
import os
import math
import shutil
import sys
import time
import traceback
from datetime import datetime

# Library imports
import geopandas as gpd
import Metashape
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon
from tqdm.auto import tqdm

# Local imports
import src.base.calc_bounds as cb
import src.base.check_sky as cs
import src.base.enhance_image as ei
import src.base.find_overlapping_images_geom as foi
import src.base.load_credentials as lc
import src.base.resize_image as rei
import src.base.rotate_image as ri
import src.dem.correct_dem as cd
import src.dem.estimate_dem_quality_old as edq
import src.export.export_pointcloud as epc
import src.export.export_thumbnail as eth
import src.export.export_tiff as eti
import src.load.load_image as li
import src.load.load_ply as lpl
import src.load.load_rema as lr
import src.load.load_rock_mask as lrm
import src.load.load_transform as lt
import src.load.load_satellite as ls
import src.other.misc.compress_tif_files as cotf
import src.pointclouds.remove_outliers as ro
import src.sfm_agi.other.create_tps_frame as ctf
import src.sfm_agi.snippets.add_gcp_markers as agm
import src.sfm_agi.snippets.align_images as ai
import src.sfm_agi.snippets.create_adapted_mask as cam
import src.sfm_agi.snippets.create_confidence_array as cca
import src.sfm_agi.snippets.create_difference_dem as cdd
import src.sfm_agi.snippets.create_matching_structure as cms  # noqa: SpellingInspection
import src.sfm_agi.snippets.create_slope_mask as csm
import src.sfm_agi.snippets.convert_ply_files as cpf
import src.sfm_agi.snippets.extract_camera_params as ecp
import src.sfm_agi.snippets.filter_markers as fm
# import src.sfm_agi.snippets.find_gcps as fg
import src.sfm_agi.snippets.find_gcps_new as fgn
import src.sfm_agi.snippets.find_tie_points_for_sfm as ftp
import src.sfm_agi.snippets.fix_ortho as fo
import src.sfm_agi.snippets.georef_ortho as go
import src.sfm_agi.snippets.georef_ortho2 as go2
import src.sfm_agi.snippets.get_project_quality as gpq
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
resolution_rel = 0.001  # in px
resolution_abs = 2  # in m
matching_method = "combined"  # which images should be matched (all, sequential, overlap)
min_overlap = 0.25  # the minimum overlap for matching with overlap
step_range = 2
custom_matching = True  # if True, custom matching will be used (lightglue)
min_tps = 15  # minimum number of tie points between images
max_tps = 10000
min_tp_confidence = 0.9
tp_type = float
min_nr_tps = 25 # 50 # for georef ortho
tp_tolerance = 0.5
custom_markers = False  # if True, custom markers support the matching of Agisoft
zoom_level_dem = 10  # in m
use_gcp_mask = True  # filter ground control points with a mask
mask_type = ["rock", "confidence", "slope"]  # "confidence" or "rock"
rock_mask_type = "REMA"
rema_level=10
mask_resolution = 10
min_gpc_tp_conf=0.75
gcp_mask_kernel_conf = 5
gcp_mask_kernel_rock = 11
min_gcp_optimum = 20
min_gcp_required = 5
min_gcp_confidence = 0.9
gcp_accuracy_px = 1 # in px
min_markers = 5
max_marker_error_px = 1 # in px
max_marker_error_m = 25 # in m
max_slope_begin = 20
max_slope_finish = 60
mask_buffer = 10  # in pixels
no_data_value = -9999
interpolate = True  # interpolate values in MESH and DEM

# Other variables
save_text_output = True  # if True, the output will be saved in a text file
save_to_psql = True
save_commands = True  # if True, the arguments of the commands will be saved in a json file
absolute_mode = True  # if False, the execution stops after the relative steps

auto_true_for_new = True  # new projects will have all steps set to True
auto_display_true_for_new = True  # new projects will have all display steps set to True
auto_debug_true_for_new = False  # new projects will have all debug steps set to True
auto_cache_true_for_new = True  # new projects will have all cache steps set to True

flag_display_steps = True  # an additional flag to enable/disable the display steps
flag_debug_steps = True  # an additional flag to enable/disable the debug steps
flag_cache_steps = True  # an additional flag to enable/disable the cache steps

# Steps
STEPS = {
    "create_masks": False,
    "union_masks": False,
    "enhance_photos": False,
    "match_photos": False,
    "align_cameras": False,
    "build_depth_maps_relative": False,
    "build_mesh_relative": False,
    "build_pointcloud_relative": False,
    "clean_pointcloud_relative": False,
    "build_dem_relative": False,
    "build_orthomosaic_relative": False,
    "build_confidence_relative": False,
    "georef_ortho": False,
    "create_gcps": True,
    "load_gcps": True,
    "filter_markers": True,
    "build_depth_maps_absolute": True,
    "save_camera_params": True,
    "build_mesh_absolute": True,
    "build_pointcloud_absolute": True,
    "clean_pointcloud_absolute": True,
    "build_dem_absolute": True,
    "build_orthomosaic_absolute": True,
    "export_alignment": True,
    "build_confidence_absolute": True,
    "build_difference_dem": True,
    "evaluate_dem": True,
    "correct_dem": True,
    "evaluate_project": False,
    "create_report": True,
    "compress_images": True,
    "fix_ortho": True,
    "copy_to_external": True,
}

DEBUG_STEPS = {
    "save_tps_to_csv": False,
    "save_transforms": False,
    "show_connections": False  # add connections lines between the images
}

# Display settings
DISPLAY_STEPS = {
    "save_thumbnails": False,
    "save_key_points": False,
    "save_tie_points": False,  # WARNING: This can be very slow
    "save_aoi": False,
}

CACHE_STEPS = {
    'save_enhanced': True,
    'use_enhanced': True,
    'save_masks': True,
    'use_masks': True,
    'save_tps': True,
    'use_tps': True,
    'save_bundler': True,
    'use_bundler': True,
    'save_rock_mask': True,
    'use_rock_mask': True,
}


def run_agi(project_name: str, images_paths: list,
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
        images_paths (list): A list of image paths.
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
        gcp_accuracy
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

    # adapt project name
    project_name = project_name.replace(" ", "_")
    project_name = project_name.lower()

    # check some combination of settings
    if resume and overwrite:
        raise ValueError("Both RESUME and OVERWRITE cannot be set to True.")
    if custom_markers and custom_matching:
        raise ValueError("Both CUSTOM_MARKERS and CUSTOM_MATCHING cannot be set to True.")
    if matching_method == "overlap" and camera_footprints is None:
        raise ValueError("Camera footprints must be provided if 'overlap' is used as matching method.")

    # set correct settings if folder not existing
    if os.path.isdir(PATH_PROJECT_FOLDERS) is False:
        print("Set resume to False and overwrite to true as fld is not existing")
        resume = False
        overwrite = True

    # set all steps to True if RESUME is True
    if resume is False and auto_true_for_new:
        print("Auto setting of all steps to True due to RESUME")
        for key in STEPS:
            STEPS[key] = True

    # set all display steps to True if RESUME is True
    if resume is False and auto_display_true_for_new:
        print("Auto setting of all display steps to True due to RESUME")
        for key in DISPLAY_STEPS:
            DISPLAY_STEPS[key] = True

    # set all debug steps to True if RESUME is True
    if resume is False and auto_debug_true_for_new:
        print("Auto setting of all debug steps to True due to RESUME")
        for key in DEBUG_STEPS:
            DEBUG_STEPS[key] = True

    # set all cache steps to True if RESUME is True
    if resume is False and auto_cache_true_for_new:
        print("Auto setting of all cache steps to True due to RESUME")
        for key in CACHE_STEPS:
            CACHE_STEPS[key] = True

    # set all display steps to False if flag_display_steps is False
    if flag_display_steps is False:
        print("Auto setting of all display steps to False")
        for key in DISPLAY_STEPS:
            DISPLAY_STEPS[key] = False

    # set all debug steps to False if flag_debug_steps is False
    if flag_debug_steps is False:
        print("Auto setting of all debug steps to False")
        for key in DEBUG_STEPS:
            DEBUG_STEPS[key] = False

    # set all cache steps to False if flag_cache_steps is False
    if flag_cache_steps is False:
        print("Auto setting of all cache steps to False")
        for key in CACHE_STEPS:
            CACHE_STEPS[key] = False

    # TEMP: set some steps to false
    print("TEMP: Disable 'save_key_points' & 'save_tie_points")
    DISPLAY_STEPS["save_key_points"] = False
    DISPLAY_STEPS["save_tie_points"] = False

    #print("TEMP: DISABLE CONFIDENCE")
    #STEPS["build_confidence_relative"] = False
    #STEPS["build_confidence_absolute"] = False

    # check if we have image
    if len(images_paths) == 0:
        raise ValueError("No images were provided")

    # check if there are duplicate entries in the images
    if len(images_paths) != len(set(images_paths)):
        raise ValueError("Duplicate entries in the images list")

    # check if the number of images is equal to the (optional) dicts
    if focal_lengths is not None and len(images_paths) != len(focal_lengths):
        raise ValueError(f"The number of images ({len(images_paths)}) "
                         f"and focal lengths ({len(focal_lengths)}) do not match.")
    if camera_footprints is not None and len(images_paths) != len(camera_footprints):
        raise ValueError(f"The number of images ({len(images_paths)}) "
                         f"and camera footprints ({len(camera_footprints)}) do not match.")
    if camera_positions is not None and len(images_paths) != len(camera_positions):
        raise ValueError(f"The number of images ({len(images_paths)}) "
                         f"and camera positions ({len(camera_positions)}) do not match.")
    if camera_accuracies is not None and len(images_paths) != len(camera_accuracies):
        raise ValueError(f"The number of images ({len(images_paths)}) "
                         f"and camera accuracies ({len(camera_accuracies)}) do not match.")
    if camera_rotations is not None and len(images_paths) != len(camera_rotations):
        raise ValueError(f"The number of images ({len(images_paths)}) "
                         f"and camera rotations ({len(camera_rotations)}) do not match.")

    # give a warning if no footprints are available
    if camera_footprints is None:
        print("WARNING: No footprints available. Only relative mode is possible.")

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

    # create a steps file
    steps_path = os.path.join(log_fld, "steps.txt")
    if os.path.exists(steps_path) is False:
        with open(steps_path, "w") as file:
            file.write(project_name + "\n")

    # create a memory file
    memory_path = os.path.join(log_fld, "memory.txt")
    if os.path.exists(memory_path) is False:
        with open(memory_path, "w") as file:
            file.write(project_name + "\n")


    try:

        # required for the safe to sfm
        status = ""
        error_raised = False
        status_message = ""

        # print a list of images
        print("Used Image ids:")
        images = [os.path.basename(image)[:-4] for image in images_paths]
        print(images)

        # print all steps
        print("STEPS:")
        print(STEPS)

        # get the license key
        licence_key = lc.load_credentials("agisoft")['licence']

        # Activate the license
        Metashape.License().activate(licence_key)

        # enable use of gpu
        Metashape.app.gpu_mask = 1
        Metashape.app.cpu_enable = False

        dem_rel = None  # relative dem
        dem_abs = None  # absolute dem
        dem_corrected = None  # corrected (absolute) dem
        dem_modern = None  # modern dem from rema
        ortho_rel = None  # relative ortho
        ortho_modern = None  # modern ortho from rema
        ortho_abs = None
        point_cloud_rel = None  # relative point cloud
        point_cloud_abs = None  # absolute point cloud
        conf_arr_rel = None  # relative confidence array
        conf_arr_abs = None  # absolute confidence array
        transform_rel = None  # the initial transform for relative mode
        transform_abs = None  # the final transform for absolute mode (improved with gcps)
        transform_modern = None  # the transform for the modern dem
        transform_corrected = None
        best_rot = None  # best rotation of georef
        rock_mask = None  # rock mask for gcps
        bounding_box_rel = None  # relative bounding box
        bounding_box_abs = None
        quality_dict = None
        quality_dict_corr = None
        proj_qual_dict = None

        # create the metashape project
        doc = Metashape.Document(read_only=False)  # noqa

        # create compression object
        compression = Metashape.ImageCompression()
        compression.tiff_compression = Metashape.ImageCompression.TiffCompressionLZW
        compression.tiff_big = True

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

        # init path variables
        output_path_dem_rel = os.path.join(output_fld,
                                           project_name + "_dem_relative.tif")
        output_path_dem_abs = os.path.join(output_fld,
                                           project_name + "_dem_absolute.tif")
        output_path_dem_corr = os.path.join(output_fld,
                                            project_name + "_dem_corrected.tif")
        output_path_conf_rel = os.path.join(output_fld,
                                            project_name + "_confidence_relative.tif")
        output_path_conf_abs = os.path.join(output_fld,
                                            project_name + "_confidence_absolute.tif")
        output_path_ortho_rel = os.path.join(output_fld,
                                             project_name + "_ortho_relative.tif")
        output_path_ortho_abs = os.path.join(output_fld,
                                             project_name + "_ortho_absolute.tif")
        output_path_pc_rel = os.path.join(output_fld,
                                          project_name + "_pointcloud_relative.ply")
        output_path_pc_abs = os.path.join(output_fld,
                                          project_name + "_pointcloud_absolute.ply")
        output_path_pc_abs_c = os.path.join(output_fld,
                                            project_name + "_pointcloud_absolute_cleaned.ply")
        output_path_diff_rela = os.path.join(output_fld,
                                             project_name + "_diff_rel.tif")
        output_path_diff_abs = os.path.join(output_fld,
                                            project_name + "_diff_abs.tif")
        output_path_diff_rela_c = os.path.join(output_fld,
                                               project_name + "_diff_rel_corrected.tif")
        output_path_diff_abs_c = os.path.join(output_fld,
                                              project_name + "_diff_abs_corrected.tif")
        transform_path = os.path.join(data_fld, "georef", "transform.txt")

        # create Interpolation object
        if interpolate:
            interpolation = Metashape.Interpolation.EnabledInterpolation
        else:
            interpolation = Metashape.Interpolation.DisabledInterpolation

        # save which images are rotated
        rotated_images = []

        # check if all images are in already in the image folder
        if len(os.listdir(img_folder)) != len(images_paths):

            start_time = time.time()

            # init variable
            img = None

            # iterate the images
            for i, image_path in (pbar := tqdm(enumerate(images_paths), total=len(images_paths))):

                pbar.set_description("Copy image to project folder")

                # get file name from path
                file_name = os.path.basename(image_path)

                pbar.set_postfix_str(f"{file_name}")

                # define output path
                output_path = os.path.join(img_folder, file_name)

                # update the image path
                images_paths[i] = output_path

                # only copy the image if it does not exist
                if os.path.exists(output_path) is False:

                    # copy the image to the output folder
                    shutil.copy(image_path, output_path)

                    # check image rotation and rotate if necessary
                    correct_rotation = cs.check_sky(output_path, conn=conn)

                    if correct_rotation is False:
                        pbar.set_postfix_str(f"{file_name} is not correctly oriented. Rotating image..")

                        # load the image
                        img = li.load_image(output_path)

                        # rotate the image
                        img = ri.rotate_image(img, 180)

                        # save the rotated image
                        eti.export_tiff(img, output_path, overwrite=True,
                                        use_lzw=True)

                        rotated_images.append(file_name[:-4])

                if i == len(images_paths) - 1:
                    pbar.set_postfix_str("Finished!")

            del img

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Copy images - finished ({exec_time:.4f} s)")

        # add a chunk for the doc and add images
        if len(doc.chunks) == 0:
            chunk = doc.addChunk()

            print("Add Photos")
            start_time = time.time()

            # add the images to the chunk
            chunk.addPhotos(images_paths, progress=_print_progress)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Add Photos - finished ({exec_time:.4f} s)")
            time.sleep(0.1)

        else:
            chunk = doc.chunks[0]

            # get all image-names from images
            image_names = [os.path.basename(image)[:-4] for image in images_paths]

            # disable all images that are not in the images list and enable all that are in
            for camera in chunk.cameras:
                if camera.label not in image_names:
                    camera.enabled = False
                else:
                    camera.enabled = True

        if DISPLAY_STEPS["save_thumbnails"]:

            # define path to the thumbnail folder
            thumb_folder = os.path.join(display_fld, "thumbnails")
            if not os.path.exists(thumb_folder):
                os.makedirs(thumb_folder)

            start_time = time.time()

            # iterate over the cameras
            for i, camera in (pbar := tqdm(enumerate(chunk.cameras), total=len(chunk.cameras))):

                pbar.set_description("Save Thumbnails")
                pbar.set_postfix_str(f"{camera.label}")

                # skip disabled cameras
                if camera.enabled is False:
                    continue

                # define path for the thumbnail
                thumb_path = os.path.join(thumb_folder, f"{camera.label}_thumb.jpg")

                # skip already existing thumbnails
                if os.path.isfile(thumb_path):
                    continue

                # load image and export as thumbnail
                image = li.load_image(camera.label)
                eth.export_thumbnail(image, thumb_path)

                # update progressbar in last loop
                if i == len(chunk.cameras) - 1:
                    pbar.set_postfix_str("Finished!")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Save Thumbnails - finished ({exec_time:.4f} s)")

        # only required for the first run
        if 'Local Coordinates' in str(chunk.crs):
            # get the image dimensions
            image_dims = {}
            for camera in chunk.cameras:

                # skip disabled cameras
                if camera.enabled is False:
                    continue

                # get the image dimensions
                image_dims[camera.label] = [camera.calibration.height, camera.calibration.width]

            # add focal length to the images
            if focal_lengths is not None:

                start_time = time.time()

                # set focal length if given
                for idx, camera in (pbar := tqdm(enumerate(chunk.cameras), total=len(chunk.cameras))):

                    pbar.set_description("Set Focal length")
                    pbar.set_postfix_str(f"{camera.label}")

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

                        # set the focal length of the camera
                        camera.sensor.focal_length = focal_length
                        camera.sensor.pixel_size = (pixel_size, pixel_size)

                        # set the fixed parameters
                        if fixed_focal_length:
                            camera.sensor.fixed_params = ['f']
                    else:
                        print(f"WARNING: Focal length not given for {camera.label}")

                    # update progressbar in last loop
                    if idx == len(chunk.cameras) - 1:
                        pbar.set_postfix_str("Finished!")

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

        counter_missing_masks = 0
        if STEPS["create_masks"] and CACHE_STEPS["use_masks"]:
            # define cache fld
            mask_cache_fld = "/data/ATM/data_1/sfm/agi_data/masks"
            mask_fld = os.path.join(data_fld, "masks_original")
            mask_adap_fld = os.path.join(data_fld, "masks_adapted")

            # create folders if required
            if not os.path.exists(mask_fld):
                os.makedirs(mask_fld)

            if not os.path.exists(mask_adap_fld):
                os.makedirs(mask_adap_fld)

            for camera in chunk.cameras:
                if camera.enabled is False:
                    continue

                # define exact cache path
                mask_cache_pth = os.path.join(mask_cache_fld, f"{camera.label}_mask.tif")

                if os.path.exists(mask_cache_pth):
                    mask = li.load_image(mask_cache_pth)

                    if mask.shape[0] != image_dims[camera.label][0] or \
                        mask.shape[1] != image_dims[camera.label][1]:
                        print("WARNING: Mask dimensions do not match the image dimensions.")
                        print(f"Mask: {mask.shape}, Image: {image_dims[camera.label]}")
                        counter_missing_masks += 1
                        continue

                    # convert the mask to a metashape image
                    mask_m = Metashape.Image.fromstring(mask,
                                                        mask.shape[1],
                                                        mask.shape[0],
                                                        channels=' ',
                                                        datatype='U8')
                    mask_obj = Metashape.Mask()
                    mask_obj.setImage(mask_m)

                    # copy mask to the mask folder
                    mask_orig_path = os.path.join(data_fld, "masks_original", f"{camera.label}_mask.tif")
                    mask_adap_path = os.path.join(data_fld, "masks_adapted", f"{camera.label}_mask.tif")
                    shutil.copy(mask_cache_pth, mask_orig_path)
                    shutil.copy(mask_cache_pth, mask_adap_path)

                    # set the mask to the camera
                    camera.mask = mask_obj

                else:
                    counter_missing_masks += 1

        print("Number of missing masks:", counter_missing_masks)

        if CACHE_STEPS["use_masks"] is False or counter_missing_masks > 0:

            if STEPS["create_masks"]:

                print("Create masks")
                start_time = time.time()

                # save step
                with open(steps_path, "a") as file:
                    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file.write(f"START: create_masks - {current_date_time}\n")

                # create temporary doc
                doc_temp = Metashape.Document(read_only=False)  # noqa

                # add a chunk for the temporary doc
                chunk_temp = doc_temp.addChunk()

                # add the images to the temporary chunk
                chunk_temp.addPhotos(images_paths, progress=_print_progress)

                # set the images to film cameras (required for finding fiducials)
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
                for idx, camera in (pbar := tqdm(enumerate(chunk_temp.cameras), total=len(chunk_temp.cameras))):

                    # skip disabled cameras
                    if camera.enabled is False:
                        continue

                    # update progressbar
                    pbar.set_description("Calibrate fiducials")
                    pbar.set_postfix_str(f"{camera.label}")

                    # calibrate the fiducials
                    try:
                        camera.sensor.calibrateFiducials(0.025)
                    except RuntimeError:
                        print(f"WARNING: Calibration of fiducials failed for {camera.label}")
                        print(f"{camera.label} will be removed from SfM processing.")
                        camera.enabled = False

                    # update progressbar in last loop
                    if idx == len(chunk_temp.cameras) - 1:
                        pbar.set_postfix_str("Finished!")

                # define path to save masks
                mask_folder = os.path.join(data_fld, "masks_original")
                if not os.path.exists(mask_folder):
                    os.makedirs(mask_folder)

                # check if the number of cameras is equal in the temporary chunk
                if len(chunk_temp.cameras) != len(chunk.cameras):
                    raise ValueError("The number of cameras in the temporary chunk is not equal to the original chunk.")

                # save masks and copy them to the original chunk as well
                for i, camera_temp in (pbar := tqdm(enumerate(chunk_temp.cameras), total=len(chunk_temp.cameras))):

                    # update description
                    pbar.set_description("Save and copy mask")
                    pbar.set_postfix_str(f"{camera_temp.label}")

                    if camera_temp.mask is not None:
                        # save the mask
                        mask_path = os.path.join(mask_folder, f"{camera_temp.label}_mask.tif")
                        camera_temp.mask.image().save(mask_path)

                        # copy the mask to the original chunk
                        camera = chunk.cameras[i]
                        camera.mask = camera_temp.mask

                    # update progressbar in last loop
                    if i == len(chunk_temp.cameras) - 1:
                        pbar.set_postfix_str("Finished!")

                # save the project
                doc.save()

                # remove the temporary doc
                del doc_temp

                # save step
                with open(steps_path, "a") as file:
                    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file.write(f"FINISHED: create_masks - {current_date_time}\n")

                finish_time = time.time()
                exec_time = finish_time - start_time
                print(f"Create masks - finished ({exec_time:.4f} s)")

            # remove all cameras that are not enabled
            for camera in chunk.cameras:
                if camera.enabled is False:
                    chunk.remove(camera)  # noqa

            # init variables
            adapted_mask, mask_bytes = None, None

            if STEPS["union_masks"]:

                time.sleep(0.1)
                start_time = time.time()

                # save step
                with open(steps_path, "a") as file:
                    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file.write(f"START: union_masks - {current_date_time}\n")

                # adapt the path to the mask folder
                mask_folder = os.path.join(data_fld, "masks_adapted")
                if not os.path.exists(mask_folder):
                    os.makedirs(mask_folder)

                # iterate over the cameras
                for idx, camera in (pbar := tqdm(enumerate(chunk.cameras), total=len(chunk.cameras))):

                    if camera.enabled is False:
                        continue

                    # update progressbar
                    pbar.set_description("Union masks")
                    pbar.set_postfix_str(f"{camera.label}")

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

                        if CACHE_STEPS["save_masks"]:
                            mask_cache_fld = "/data/ATM/data_1/sfm/agi_data/masks"
                            eti.export_tiff(adapted_mask, os.path.join(mask_cache_fld, f"{camera.label}_mask.tif"),
                                            use_lzw=True, overwrite=True)

                        # create a mask object
                        mask_obj = Metashape.Mask()
                        mask_obj.setImage(adapted_mask_m)

                        # set the mask to the camera
                        camera.mask = mask_obj

                        # save the adapted mask
                        mask_path = os.path.join(mask_folder, f"{camera.label}_mask.tif")
                        camera.mask.image().save(mask_path)

                    # update progressbar in last loop
                    if idx == len(chunk.cameras) - 1:
                        pbar.set_postfix_str("Finished!")

                # save step
                with open(steps_path, "a") as file:
                    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file.write(f"FINISHED: union_masks - {current_date_time}\n")

                finish_time = time.time()
                exec_time = finish_time - start_time
                print(f"Union masks - finished ({exec_time:.4f} s)")

                # save the project
                doc.save()

            # reset variable
            del adapted_mask
            del mask_bytes

        # init variable for enhanced folder
        enhanced_folder = os.path.join(data_fld, "images_enhanced")

        # enhance the images
        if STEPS["enhance_photos"]:

            time.sleep(0.1)
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: enhance_photos - {current_date_time}\n")

            # create folder for the enhanced images
            if not os.path.exists(enhanced_folder):
                os.makedirs(enhanced_folder)

            # iterate over the cameras
            for idx, camera in (pbar := tqdm(enumerate(chunk.cameras), total=len(chunk.cameras))):

                if camera.enabled is False:
                    continue

                pbar.set_description("Enhance photos")
                pbar.set_postfix_str(f"{camera.label}")

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

                # define path for cache
                enhanced_fld = "/data/ATM/data_1/sfm/agi_data/enhanced"
                enhanced_cache_pth = os.path.join(enhanced_fld, f"{camera.label}.tif")

                # get the enhanced image if existing
                if CACHE_STEPS['use_enhanced']:
                    if os.path.exists(enhanced_cache_pth):
                        enhanced_image = li.load_image(enhanced_cache_pth)
                        if enhanced_image.shape != image.shape:
                            enhanced_image = ei.enhance_image(image, mask)
                    else:
                        enhanced_image = ei.enhance_image(image, mask)
                else:
                    enhanced_image = ei.enhance_image(image, mask)

                # save the enhanced image to the cache folder
                if CACHE_STEPS['save_enhanced'] and not os.path.exists(enhanced_cache_pth):
                    eti.export_tiff(enhanced_image, enhanced_cache_pth,
                                    overwrite=True, use_lzw=True)

                # save the enhanced image to the enhanced folder as well
                eti.export_tiff(enhanced_image, e_image_pth,
                                overwrite=True, use_lzw=True)

                # update the image in the chunk
                photo = camera.photo.copy()
                photo.path = e_image_pth
                camera.photo = photo

                # update progressbar in last loop
                if idx == len(chunk.cameras) - 1:
                    pbar.set_postfix_str("Finished!")

            del mask_bytes
            del enhanced_image

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: enhance_photos - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Enhance photos - finished ({exec_time:.4f} s)")

            # save the project
            doc.save()

        # save thumbnails of the enhanced images
        if DISPLAY_STEPS["save_thumbnails"] and STEPS["enhance_photos"]:

            # create folder for the thumbnails
            thumb_folder = os.path.join(display_fld, "thumbnails")
            if not os.path.exists(thumb_folder):
                os.makedirs(thumb_folder)

            time.sleep(0.1)
            start_time = time.time()

            for idx, camera in (pbar := tqdm(enumerate(chunk.cameras), total=len(chunk.cameras))):

                if camera.enabled is False:
                    continue

                pbar.set_description("Save Thumbnails (enhanced)")
                pbar.set_postfix_str(f"{camera.label}")

                img_path = os.path.join(data_fld, "images_enhanced", f"{camera.label}.tif")
                thumb_path = os.path.join(thumb_folder, f"{camera.label}_e_thumb.jpg")

                image = li.load_image(img_path)
                eth.export_thumbnail(image, thumb_path)

                # update progressbar in last loop
                if idx == len(chunk.cameras) - 1:
                    pbar.set_postfix_str("Finished!")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Save Thumbnails (enhanced) - finished ({exec_time:.4f} s)")

        # match photos
        if STEPS["match_photos"]:

            print("Match photos")
            time.sleep(0.1)
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: match_photos - {current_date_time}\n")

            if custom_matching:

                bundler_folder_path = os.path.join(data_fld, "bundler")
                if not os.path.exists(bundler_folder_path):
                    os.mkdir(bundler_folder_path)

                # define path to the bundler file
                bundler_path = os.path.join(bundler_folder_path, "bundler.out")

                # per default, we calculate the bundler file new
                used_cached_bundler = False

                if CACHE_STEPS['use_bundler']:

                    # define path for existing bundler files and the txt files
                    new_bundler_fld = "/data/ATM/data_1/sfm/agi_data/bundler_files/"
                    new_bundler_path = os.path.join(new_bundler_fld,
                                                    f"{project_name}_{len(chunk.cameras)}.out")
                    new_bundler_txt_path = os.path.join(new_bundler_fld,
                                                        f"{project_name}_{len(chunk.cameras)}_cameras.txt")

                    # check if the bundler file is existing
                    if os.path.exists(new_bundler_path):

                        # read the txt file and check if the cameras and chunk are identical
                        with open(new_bundler_txt_path, "r") as f:
                            new_camera_names = f.readlines()
                            new_camera_names = [name.strip() for name in new_camera_names]

                        # get all camera names
                        camera_names = [camera.label for camera in chunk.cameras if camera.enabled]
                        camera_names.sort()

                        # check if the camera names are identical
                        if camera_names == new_camera_names:
                            print(new_bundler_path, bundler_path)
                            shutil.copy(new_bundler_path, bundler_path)
                            used_cached_bundler = True

                # we need to calculate the bundler file new
                if used_cached_bundler is False:
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
                                                                     image_dims,
                                                                     mask_folder=mask_folder,
                                                                     matching_method=matching_method,
                                                                     footprint_dict=camera_footprints,
                                                                     rotation_dict=camera_rotations,
                                                                     tp_type=tp_type,
                                                                     min_overlap=min_overlap,
                                                                     step_range=step_range,
                                                                     min_conf=min_tp_confidence,
                                                                     min_tps=min_tps,
                                                                     max_tps=max_tps,
                                                                     use_cached_tps=CACHE_STEPS['use_tps'],
                                                                     save_tps=CACHE_STEPS['save_tps'])

                    # create the ply files for the custom matching
                    cms.create_matching_structure(tp_dict, conf_dict,
                                                  project_files_path,
                                                  img_folder,
                                                  tolerance=tp_tolerance)

                    if CACHE_STEPS['save_bundler']:
                        # define new path for bundler file
                        new_bundler_fld = "/data/ATM/data_1/sfm/agi_data/bundler_files/"
                        new_bundler_path = os.path.join(new_bundler_fld,
                                                        f"{project_name}_{len(chunk.cameras)}.out")
                        # copy the bundler file to the new folder
                        shutil.copy(bundler_path, new_bundler_path)

                        # create a txt file with the camera names alphabetically sorted
                        camera_names = [camera.label for camera in chunk.cameras if camera.enabled]
                        camera_names.sort()
                        with open(os.path.join(new_bundler_fld,
                                               f"{project_name}_{len(chunk.cameras)}_cameras.txt"), "w") as f:
                            for name in camera_names:
                                f.write(f"{name}\n")

                # import the bundler file
                chunk.importCameras(bundler_path, format=Metashape.CamerasFormatBundler)

            else:

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

                    if camera_footprints is None:
                        raise ValueError("Camera footprints must be provided if 'overlap' is used as matching method.")

                    # get all camera labels as image ids
                    image_ids = [camera.label for camera in chunk.cameras if camera.enabled]

                    # get the footprints as lst
                    footprints_lst = [camera_footprints[image_id] for image_id in image_ids]

                    overlap_dict = foi.find_overlapping_images_geom(image_ids,
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

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: match_photos - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Match photos - finished ({exec_time:.4f} s)")

            # save the project
            doc.save()

        # just to have a block for resetting "enhance images"
        if True:
            # set images back to the original images
            time.sleep(0.1)
            start_time = time.time()

            # Initialize progress bar as None
            pbar = None

            # check if we update a picture (for later saving of the project)
            pic_updated = False

            # iterate over the cameras
            for i, camera in enumerate(chunk.cameras):

                # skip disabled cameras
                if camera.enabled is False:
                    continue

                # get original image path
                image_pth = os.path.join(img_folder, f"{camera.label}.tif")

                # check if camera has original path
                if camera.photo.path != image_pth:

                    # Initialize progress bar only if it hasn’t been created yet
                    if pbar is None:
                        pbar = tqdm(total=len(chunk.cameras))

                    pbar.set_description("Restore original image")
                    pbar.set_postfix_str(f"{camera.label}")

                    # update the image in the chunk
                    photo = camera.photo.copy()
                    photo.path = image_pth
                    camera.photo = photo

                    # set flag to true
                    pic_updated = True

                if i == len(chunk.cameras) - 1:
                    if pbar is not None:
                        pbar.set_postfix_str("Finished!")

            # finish the progress bar and save the project
            if pic_updated:
                finish_time = time.time()
                exec_time = finish_time - start_time
                print(f"Restore original images - finished ({exec_time:.4f} s)")

                # save the project
                doc.save()

            # reset the progress bar
            pbar = None

        # align cameras
        if STEPS["align_cameras"]:

            print("Align cameras")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: align_cameras - {current_date_time}\n")

            arguments = {
                'reset_alignment': False,
                'adaptive_fitting': True
            }

            # we need to reset the alignment if we apply custom matching
            if custom_matching:
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

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: align_cameras - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Align cameras - finished ({exec_time:.4f} s)")

        # save key points
        if DISPLAY_STEPS["save_key_points"]:

            print("Save key points")
            start_time = time.time()

            # define point cloud folder
            point_cloud_fld = os.path.join(project_files_path, "0", "0", "point_cloud")

            # check if this folder exists
            if os.path.exists(point_cloud_fld) is False:
                raise FileNotFoundError("Point cloud folder does not exist. Please align the images first.")

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

        # save the tie points to a csv file
        if DEBUG_STEPS["save_tps_to_csv"]:

            # get path to point cloud fld
            point_cloud_fld = os.path.join(project_files_path, "0", "0", "point_cloud")
            tracks_path = os.path.join(point_cloud_fld, "point_cloud", "tracks.txt")

            # check if this folder exists
            if os.path.exists(point_cloud_fld) is False:
                raise FileNotFoundError("Point cloud folder does not exist. Please align the images first.")

            # check if the folder is already unzipped (with tracks.ply)
            if os.path.exists(tracks_path) is False:
                # convert ply files to txt files
                cpf.convert_ply_files(point_cloud_fld, True)

            # create the dataframe with tie_point/tracks
            path_fld_points = os.path.join(point_cloud_fld, "point_cloud")
            tps_df = ctf.create_tps_frame(path_fld_points)

            output_pth = os.path.join(debug_fld, "tie_points.csv")
            tps_df.to_csv(output_pth, index=False)

        # save tie points
        if DISPLAY_STEPS["save_tie_points"]:

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

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_depth_maps_relative - {current_date_time}\n")

            arguments = {}

            chunk.buildDepthMaps(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildDepthMaps_relative", arguments, argument_fld)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_depth_maps_relative - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative depth maps - finished ({exec_time:.4f} s)")

            # save the project
            doc.save()

        # relative bounding box only required for local crs
        if 'Local Coordinates' in str(chunk.crs) or STEPS["create_gcps"]:
            print("Create relative bounding box")
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

            # define a bounding box for the relative coords
            if camera_positions is not None:
                min_corner = chunk.crs.project(chunk.transform.matrix.mulp(min_corner))
                max_corner = chunk.crs.project(chunk.transform.matrix.mulp(max_corner))

            # get height and width of the bounding box area
            print("Size of the bounding box:")
            print(f"Width: {max_corner.x - min_corner.x}")
            print(f"Height: {max_corner.y - min_corner.y}")

            # restrain the bounding box to the absolute bounds if given
            """
            if camera_positions is not None and absolute_bounds is not None:
                min_corner.x = max(min_corner.x, absolute_bounds[0])
                min_corner.y = max(min_corner.y, absolute_bounds[1])
                max_corner.x = min(max_corner.x, absolute_bounds[2])
                max_corner.y = min(max_corner.y, absolute_bounds[3])
            """

            # create 2d versions of the corners
            min_corner_2d = Metashape.Vector([min_corner.x, min_corner.y])  # noqa
            max_corner_2d = Metashape.Vector([max_corner.x, max_corner.y])  # noqa

            # Create the bounding box
            bounding_box_rel = Metashape.BBox(min_corner_2d, max_corner_2d)  # noqa

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

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_mesh_relative - {current_date_time}\n")

            arguments = {
                'surface_type': Metashape.Arbitrary,
                'interpolation': Metashape.EnabledInterpolation,
            }

            # build mesh
            chunk.buildModel(**arguments)
            doc.save()

            print(" Export mesh to file")

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

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_mesh_relative - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative mesh - finished ({exec_time:.4f} s)")

        # build point cloud
        if STEPS["build_pointcloud_relative"]:

            print("Build relative point cloud")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_pointcloud_relative - {current_date_time}\n")

            arguments = {
                'point_colors': True,
                'point_confidence': True
            }

            chunk.buildPointCloud(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildPointCloud_relative",
                                   arguments, argument_fld)

            # save the project
            doc.save()

            print(" Export point cloud to file")

            # export parameters for the point cloud
            arguments = {
                'path': output_path_pc_rel,
                'save_point_color': True,
                'save_point_confidence': True
            }

            # export the point cloud
            chunk.exportPointCloud(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("exportPointCloud_relative",
                                   arguments, argument_fld)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_pointcloud_relative - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative point cloud - finished ({exec_time:.4f} s)")

        # build DEM
        if STEPS["build_dem_relative"]:

            print("Build relative DEM")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_dem_relative - {current_date_time}\n")

            arguments = {
                'source_data': Metashape.DataSource.PointCloudData,
                'interpolation': Metashape.Interpolation.EnabledInterpolation,
            }

            # add region to build parameters
            if bounding_box_rel is not None:
                arguments['region'] = bounding_box_rel

            # add resolution to build parameters dependent on camera positions
            if camera_positions is None:
                arguments['resolution'] = resolution_rel  # noqa
            else:
                arguments['resolution'] = resolution_abs

            # build the DEM
            chunk.buildDem(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildDem_relative",
                                   arguments, argument_fld)

            # save the project
            doc.save()

            print(" Export DEM to file")

            # set export parameters for the DEM
            arguments = {
                'path': output_path_dem_rel,
                'source_data': Metashape.ElevationData,
                'image_format': Metashape.ImageFormatTIFF,
                'raster_transform': Metashape.RasterTransformNone,
                'nodata_value': no_data_value,
                'image_compression': compression,
                'clip_to_boundary': True,
            }

            # add region to export parameters
            if bounding_box_rel is not None:
                arguments['region'] = bounding_box_rel

            # add resolution to build parameters dependent on camera positions
            if camera_positions is None:
                arguments['resolution_x'] = resolution_rel  # noqa
                arguments['resolution_y'] = resolution_rel  # noqa
            else:
                arguments['resolution_x'] = resolution_abs
                arguments['resolution_y'] = resolution_abs

            # export the DEM
            chunk.exportRaster(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("exportDEM_relative",
                                   arguments, argument_fld)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_dem_relative - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative DEM - finished ({exec_time:.4f} s)")

        # build ortho-mosaic
        if STEPS["build_orthomosaic_relative"]:

            print("Build relative orthomosaic")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_orthomosaic_relative - {current_date_time}\n")

            # arguments for building the orthomosaic
            arguments = {
                'surface_data': Metashape.ModelData,
                'blending_mode': Metashape.MosaicBlending,
            }

            # add region to build parameters
            if bounding_box_rel is not None:
                arguments['region'] = bounding_box_rel

            # add resolution to build parameters dependent on camera positions
            if camera_positions is None:
                arguments['resolution_x'] = resolution_rel  # noqa
                arguments['resolution_y'] = resolution_rel  # noqa
            else:
                arguments['resolution_x'] = resolution_abs
                arguments['resolution_y'] = resolution_abs

            # build the orthomosaic
            chunk.buildOrthomosaic(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildOrthomosaic_relative",
                                   arguments, argument_fld)

            # save the project
            doc.save()

            print(" Export orthomosaic to file")

            # set export parameters for the orthomosaic
            arguments = {
                'path': output_path_ortho_rel,
                'source_data': Metashape.OrthomosaicData,
                'image_format': Metashape.ImageFormatTIFF,
                'raster_transform': Metashape.RasterTransformNone,
                # 'nodata_value': no_data_value, only for DEMs
                'image_compression': compression,
                'clip_to_boundary': True,
            }

            # add region to export parameters
            if bounding_box_rel is not None:
                arguments['region'] = bounding_box_rel

            # add resolution to build parameters dependent on camera positions
            if camera_positions is None:
                arguments['resolution'] = resolution_rel  # noqa
            else:
                arguments['resolution'] = resolution_abs

            # export the orthomosaic
            chunk.exportRaster(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("exportOrthomosaic_relative",
                                   arguments, argument_fld)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_orthomosaic_relative - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative orthomosaic - finished ({exec_time:.4f} s)")

        # save memory consumption
        """
        import sys
        def sizeof_fmt(num, suffix='B'):
            ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
            for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
                if abs(num) < 1024.0:
                    return "%3.1f %s%s" % (num, unit, suffix)
                num /= 1024.0
            return "%.1f %s%s" % (num, 'Yi', suffix)

        with open(memory_path, "a") as file:
            for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                    locals().items())), key=lambda x: -x[1])[:20]:
                file.write("{:>30}: {:>8}\n".format(name, sizeof_fmt(size)))
                print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        """

        # build confidence array
        if STEPS["build_confidence_relative"]:
            print("Build relative confidence array")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_confidence_relative - {current_date_time}\n")

            # load the dem
            if dem_rel is None:
                dem_rel, transform_rel = li.load_image(output_path_dem_rel,
                                                       return_transform=True)

            # load the point cloud
            if point_cloud_rel is None:
                point_cloud_rel = lpl.load_ply(output_path_pc_rel)

            # create the confidence array
            conf_arr_rel = cca.create_confidence_arr(dem_rel,
                                                     point_cloud_rel,
                                                     transform_rel,
                                                     interpolate=True,
                                                     distance=10)

            # export the confidence array
            eti.export_tiff(conf_arr_rel, output_path_conf_rel,
                            transform=transform_rel, overwrite=True, use_lzw=True)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_confidence_relative - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build relative confidence array - finished ({exec_time:.4f} s)")

        # stop if we don't have absolute data
        if absolute_mode is False or camera_footprints is None:
            print("Finished processing relative data & absolute mode is disabled")
            return

        # georeference the ortho
        if STEPS["georef_ortho"]:

            print("Georeference ortho-photo")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: georef_ortho - {current_date_time}\n")

            if os.path.exists(output_path_ortho_rel) is False:
                raise FileNotFoundError(f"Ortho file does not exist "
                                        f"at '{output_path_ortho_rel}'")

            # load the required data
            if ortho_rel is None:
                ortho_rel = li.load_image(output_path_ortho_rel)
            if camera_footprints is None:
                camera_footprints = {}

            # if ortho has alpha -> remove it
            if len(ortho_rel.shape) == 3:
                ortho_rel = ortho_rel[0, :, :]

            # check which cameras are aligned
            image_ids = []
            aligned = []
            for camera in chunk.cameras:

                image_ids.append(camera.label)

                if camera.enabled is False:
                    continue

                if camera.transform:
                    aligned.append(True)
                else:
                    aligned.append(False)

            # create path for the geo-referenced fld
            georef_data_fld = os.path.join(data_fld, "georef")
            if os.path.isdir(georef_data_fld) is False:
                os.makedirs(georef_data_fld)

            # create path for the tps
            ortho_tps_path = os.path.join(georef_data_fld,
                                          "ortho_tps.csv")

            # create path for the ortho
            ortho_georef_path = os.path.join(georef_data_fld,
                                             "ortho_georeferenced.tif")

            # create path for best rotation
            rot_path = os.path.join(georef_data_fld, "best_rot.txt")

            # georef with complete autorotate
            tpl = go.georef_ortho(ortho_rel,
                                  image_ids,
                                  list(camera_footprints.values()),
                                  aligned,
                                  azimuth=None, auto_rotate=True,
                                  trim_image=True,
                                  min_nr_tps=min_nr_tps,
                                  tp_type=tp_type,
                                  min_conf=0.5,
                                  start_conf=0.9,
                                  only_vertical_footprints=True,
                                  save_path_tps=ortho_tps_path,
                                  save_path_ortho=ortho_georef_path,
                                  save_path_rot=rot_path,
                                  save_path_transform=transform_path)
            transform_georef = tpl[0]
            bounds_georef = tpl[1]
            best_rot = tpl[2]
            num_tps = tpl[3]

            if transform_georef is None and absolute_bounds is not None:
                print("Try georef with absolute bounds")
                tpl = go.georef_ortho(ortho_rel,
                                      image_ids,
                                      list(camera_footprints.values()),
                                      aligned,
                                      sat_bounds = absolute_bounds,
                                      azimuth=None, auto_rotate=True,
                                      trim_image=True,
                                      min_nr_tps=min_nr_tps,
                                      tp_type=tp_type,
                                      min_conf=0.5,
                                      start_conf=0.9,
                                      only_vertical_footprints=True,
                                      save_path_tps=ortho_tps_path,
                                      save_path_ortho=ortho_georef_path,
                                      save_path_rot=rot_path,
                                      save_path_transform=transform_path)
                transform_georef = tpl[0]
                bounds_georef = tpl[1]
                best_rot = tpl[2]
                num_tps = tpl[3]

            if best_rot is None:
                raise ValueError("No tps found at all. Check the georef_ortho function.")

            if transform_georef is None:
                print("No transformation found with auto-rotation. Trying with georef_ortho2")

            georef_2 = True
            if georef_2:
                if os.path.exists(ortho_georef_path):
                    gref_ortho = li.load_image(ortho_georef_path)
                else:
                    gref_ortho = ortho_rel

                t_g, desc = go2.georef_ortho_2(gref_ortho,
                                               transform_georef, best_rot,
                                               bounds_georef,
                                               min_nr_tps=min_nr_tps,
                                               tps_to_beat=num_tps,
                                               tp_type=tp_type,
                                               save_path_tps=ortho_tps_path,
                                               save_path_ortho=ortho_georef_path,
                                               save_path_transform=transform_path)

                if desc == "too_few_tps":
                    print(f"Not enough tie points found in go2")
                    raise ValueError("Transform is not defined. Check the georef_ortho2 function.")
                elif desc == "no_improvement":
                    print("No improvement in the transformation")
                elif desc == "success":
                    transform_georef = t_g

            if transform_georef is None:
                raise ValueError("Transform is not defined. Check the georef_ortho function.")

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: georef_ortho - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Georeference ortho-photo - finished ({exec_time:.4f} s)")

        if STEPS["create_gcps"]:

            print("Create GCPs")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: create_gcps - {current_date_time}\n")

            # set expected accuracy of the markers in px
            chunk.marker_projection_accuracy = gcp_accuracy_px

            # define output path in which gcp files are saved
            georef_fld = os.path.join(data_fld, "georef")
            if not os.path.exists(georef_fld):
                os.makedirs(georef_fld)

            # load dem if not loaded yet
            if dem_rel is None:
                print("  Load DEM")
                dem_rel = li.load_image(output_path_dem_rel)

            # load ortho if not loaded yet
            if ortho_rel is None:
                print("  Load Ortho")
                ortho_rel = li.load_image(output_path_ortho_rel)
                # remove alpha channel
                if len(ortho_rel.shape) == 3:
                    ortho_rel = ortho_rel[0, :, :]

            if best_rot is None:
                georef_data_fld = os.path.join(data_fld, "georef")
                rot_path = os.path.join(georef_data_fld, "best_rot.txt")
                best_rot = np.loadtxt(rot_path)
                best_rot = float(best_rot)

            # load transform and bounds for geo-referencing
            transform_georef = lt.load_transform(transform_path, delimiter=",")
            bounds_georef_old = cb.calc_bounds(transform_georef, dem_rel.shape)

            min_x = bounds_georef_old[0]
            min_y = bounds_georef_old[1]
            max_x = bounds_georef_old[2]
            max_y = bounds_georef_old[3]

            # round values to the next step of 10
            min_x_r = math.ceil(min_x / 10) * 10
            min_y_r = math.ceil(min_y / 10) * 10
            max_x_r = math.floor(max_x / 10) * 10
            max_y_r = math.floor(max_y / 10) * 10

            # calculate new bounds
            bounds_georef = [min_x_r, min_y_r, max_x_r, max_y_r]

            # get difference in min_x and min_y
            diff_min_x = round(min_x_r - min_x, 2)
            diff_min_y = round(min_y_r - min_y, 2)
            diff_max_x = round(max_x_r - max_x, 2)
            diff_max_y = round(max_y_r - max_y, 2)
            diffs = (diff_min_x, diff_min_y, diff_max_x, diff_max_y)

            print("Old bounds:", bounds_georef_old)
            print("New bounds:", bounds_georef)
            print("Differences:", diffs)

            # load modern dem
            print("  Load modern DEM")
            dem_modern, transform_rema = lr.load_rema(bounds_georef,
                                                      zoom_level=rema_level,
                                                      auto_download=True,
                                                      return_transform=True)

            # load modern ortho
            print("  Load modern Ortho")
            ortho_modern, transform_modern = ls.load_satellite(bounds_georef,
                                                               return_transform=True)

            # assure that the transforms are identical
            if transform_rema != transform_modern:
                raise ValueError("Transforms of DEM and Ortho are not identical")

            # set path to gcps
            gcp_path = os.path.join(georef_fld, "gcps.csv")

            if bounding_box_rel is None:
                raise ValueError("Bounding box is not defined. "
                                 "Please create a bounding box first.")

            # add resolution to build parameters dependent on camera positions
            if camera_positions is None:
                resolution = resolution_rel
            else:
                resolution = resolution_abs

            # create optional mask
            if use_gcp_mask:

                # define base mask (where nothing is masked)
                mask_modern = np.ones(ortho_modern.shape[-2:], dtype=bool)
                mask_rel = np.ones(dem_rel.shape, dtype=bool)

                # add rock mask
                if "rock" in mask_type:

                    print("  Load rock mask for", bounds_georef)

                    path_cached_rock_mask = os.path.join(georef_fld, "rock_mask.tif")

                    if CACHE_STEPS["use_rock_mask"]:
                        if os.path.isfile(path_cached_rock_mask):
                            print("Load cached rock mask")
                            rock_mask = li.load_image(path_cached_rock_mask)
                        else:
                            rock_mask = None

                    # calc the rock mask
                    if rock_mask is None or rock_mask.shape != mask_modern.shape:
                        print("  Create rock mask")
                        rock_mask = lrm.load_rock_mask(bounds_georef,
                                                       mask_resolution,
                                                       mask_buffer=gcp_mask_kernel_rock)

                    if CACHE_STEPS["save_rock_mask"] and rock_mask is not None:
                        eti.export_tiff(rock_mask, path_cached_rock_mask,
                                        overwrite=True, use_lzw=True)

                    # give a warning if no rocks are existing in the mask
                    if np.sum(rock_mask) == 0:
                        print("WARNING: No rocks found in the rock mask")

                    # adapt mask based on rock mask
                    mask_modern[rock_mask == 0] = 0

                # add confidence mask
                if "confidence" in mask_type:

                    print("  Create confidence mask")

                    # load the confidence array
                    if conf_arr_rel is None:
                        try:
                            conf_arr_rel = li.load_image(output_path_conf_rel)
                            conf_arr_rel[np.isnan(conf_arr_rel)] = 0
                        except:
                            conf_arr_rel = None

                    # check if the confidence array is existing
                    if conf_arr_rel is None:
                        raise ValueError("Confidence array is not defined. "
                                         "Please create a confidence array first.")


                    # adapt mask based on confidence
                    mask_rel[conf_arr_rel < min_gcp_confidence] = 0
                    mask_rel[conf_arr_rel >= min_gcp_confidence] = 1

                    # apply kernel to mask
                    mask_rel = mask_rel.astype(np.uint8)
                    kernel = np.ones((gcp_mask_kernel_conf, gcp_mask_kernel_conf), np.uint8)
                    mask_rel = cv2.dilate(mask_rel, kernel, iterations=1)
                    mask_rel = mask_rel.astype(bool)

                # add slope mask
                if "slope" in mask_type:

                    print("  Create slope mask")

                    global max_slope_begin
                    base_mask_modern = mask_modern.copy()

                    mask_modern = csm.create_slope_mask(dem_modern, transform_rema,
                                                        mask_modern, max_slope_begin)

                # check if nothing is masked -> remove mask
                if np.all(mask_modern == 1):
                    mask_modern = None
                if np.all(mask_rel == 1):
                    mask_rel = None
            else:
                # no masks are used
                mask_modern = None
                mask_rel = None

            print("  Find GCPs")

            # call snippet to export gcps
            gcp_df = fgn.find_gcps_new(dem_rel, dem_modern,
                                  ortho_rel, ortho_modern,
                                  transform_modern,
                                  resolution, bounding_box_rel,
                                  bounds=absolute_bounds,
                                  rotation=best_rot,
                                  min_conf=min_gpc_tp_conf,
                                  mask_old=mask_rel, mask_new=mask_modern,
                                  raise_error=False)

            if "slope" in mask_type:

                while max_slope_begin <= max_slope_finish:

                    # we found enough tie-points
                    if gcp_df.shape[0] >= min_gcp_optimum:
                        break

                    # increase the slope
                    max_slope_begin = max_slope_begin + 5


                    # adapt new mask
                    print(f"Create slope mask with {max_slope_begin}")
                    mask_modern = base_mask_modern.copy()

                    mask_modern = csm.create_slope_mask(dem_modern, transform_rema,
                                                        mask_modern, max_slope_begin)

                    gcp_df = fgn.find_gcps_new(dem_rel, dem_modern,
                                          ortho_rel, ortho_modern,
                                          transform_modern,
                                          resolution, bounding_box_rel,
                                          rotation=best_rot,
                                          min_conf=min_gpc_tp_conf,
                                          mask_old=mask_rel, mask_new=mask_modern,
                                          raise_error=False)

            if gcp_df.shape[0] < min_gcp_required:
                raise ValueError("Too few GCPs are found. Please check the orthophoto or "
                                 "increase the mask buffer.")


            # Create labels (n x 1 array)
            num_points = gcp_df.shape[0]
            labels = pd.Series([f"gcp_{i + 1}" for i in range(num_points)])

            gcp_df.insert(0, 'GCP', labels)

            gcp_df.to_csv(gcp_path, sep=';', index=False, float_format='%.8f')

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: create_gcps - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Create GCPs - finished ({exec_time:.4f} s)")

            # delete the masks
            mask_modern = None
            mask_rel = None
            base_mask_modern = None
            del mask_modern, mask_rel, base_mask_modern

        if STEPS["load_gcps"]:

            print("Load GCPs")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: load_gcps - {current_date_time}\n")

            # set path to gcps
            gcp_path = os.path.join(data_fld, "georef", "gcps.csv")

            # check if we have a gcp file
            if os.path.isfile(gcp_path) is False:
                raise FileNotFoundError(f"GCP file does not exist at '{gcp_path}'")

            # load the gcps
            gcps = pd.read_csv(gcp_path, sep=';')

            print(f"{gcps.shape[0]} GCPS are loaded from file")

            if gcps.shape[0] == 0:
                raise ValueError("No GCPs are loaded. Check the previous steps for completeness.")

            # add markers to the chunk
            agm.add_gcp_markers(chunk, gcps, accuracy=gcp_accuracy,
                                epsg_code=epsg_code, reset_markers=True)

            # "https://www.agisoft.com/forum/index.php?topic=7446.0"
            # "https://www.agisoft.com/forum/index.php?topic=10855.0"
            doc.save()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: load_gcps - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Load GCPs - finished ({exec_time:.4f} s)")

        if STEPS["export_alignment"]:

            print("Export alignment")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: export_alignment - {current_date_time}\n")

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

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: export_alignment - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Export alignment - finished ({exec_time:.4f} s)")

        # check if chunk is already in absolute mode
        if 'Local Coordinates' in str(chunk.crs):
            print("Set to absolute mode")
            start_time = time.time()

            # set chunk crs to absolute mode
            chunk.crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa
            chunk.camera_crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa

            # save the project
            doc.save()

            # update alignment and transform
            chunk.optimizeCameras()
            chunk.updateTransform()

            # save the project
            doc.save()

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Set to absolute mode - finished ({exec_time:.4f} s)")

        # set the projection to the crs
        projection_abs = Metashape.OrthoProjection()
        projection_abs.crs = chunk.crs

        # remove some markers
        if STEPS["filter_markers"]:

            print("Filter Markers")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: filter_markers - {current_date_time}\n")

            # filter gcps
            fm.filter_markers(chunk, min_markers,
                              max_marker_error_px, max_marker_error_m,
                              delete=False)

            # save the project
            doc.save()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: filter_markers - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Filter Markers - finished ({exec_time:.4f} s)")

        if STEPS["build_depth_maps_absolute"]:

            print("Build absolute depth maps")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_depth_maps_absolute - {current_date_time}\n")

            # delete all existing depth maps
            for dm in chunk.depth_maps_sets:
                chunk.remove(dm)  # noqa
            doc.save()

            # arguments for building the depth maps
            arguments = {}

            # build depth maps
            chunk.buildDepthMaps(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildDepthMaps_absolute",
                                   arguments, argument_fld)

            # save the project
            doc.save()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_depth_maps_absolute - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute depth maps - finished ({exec_time:.4f} s)")

        # just to have a block for "build absolute bounding box"
        if True:
            print("Create absolute bounding box")
            start_time = time.time()

            # get attributes of the chunk
            center = chunk.region.center
            size = chunk.region.size
            rotate = chunk.region.rot
            transform = chunk.transform.matrix

            # Calculate corners of the bounding box
            corners = [transform.mulp(center + rotate * Metashape.Vector(  # noqa
                [size[0] * ((i & 1) - 0.5), 0.5 * size[1] * ((i & 2) - 1),  # noqa
                 0.25 * size[2] * ((i & 4) - 2)])) for i in  # noqa
                       range(8)]  # noqa

            # make corners absolute
            if chunk.crs:
                corners = [chunk.crs.project(x) for x in corners]

            min_x = min(corners, key=lambda x: x[0])[0]  # noqa
            max_x = max(corners, key=lambda x: x[0])[0]  # noqa
            min_y = min(corners, key=lambda x: x[1])[1]  # noqa
            max_y = max(corners, key=lambda x: x[1])[1]  # noqa

            print("Absolute bounds", absolute_bounds)

            # restrain the bounding box to the absolute bounds if given
            if absolute_bounds is not None:
                min_x = max(min_x, absolute_bounds[0])
                min_y = max(min_y, absolute_bounds[1])
                max_x = min(max_x, absolute_bounds[2])
                max_y = min(max_y, absolute_bounds[3])


            # round values to the next step of 10
            min_x = math.ceil(min_x / 10) * 10
            min_y = math.ceil(min_y / 10) * 10
            max_x = math.floor(max_x / 10) * 10
            max_y = math.floor(max_y / 10) * 10

            # set bounding box to int values
            min_x = int(min_x)
            min_y = int(min_y)
            max_x = int(max_x)
            max_y = int(max_y)

            # get height and width of the bounding box
            width = np.abs(max_x - min_x)
            height = np.abs(max_y - min_y)

            # check if the resolution fits in the bounding box
            if width % resolution_abs != 0:
                width = width - (width % resolution_abs)
                max_x = min_x + width
            if height % resolution_abs != 0:
                height = height - (height % resolution_abs)
                max_y = min_y + height

            # create 2d vectors
            min_corner_2d = Metashape.Vector([min_x, min_y])  # noqa
            max_corner_2d = Metashape.Vector([max_x, max_y])  # noqa

            # Create the bounding box
            bounding_box_abs = Metashape.BBox(min_corner_2d, max_corner_2d)  # noqa

            # get corners of the bounding box
            print("Corners of the absolute bounding box:")
            print(f"Min: {min_corner_2d}")
            print(f"Max: {max_corner_2d}")

            # get height and width of the bounding box area
            print("Size of the bounding box:")
            bbox_width = max_corner_2d.x - min_corner_2d.x
            bbox_height = max_corner_2d.y - min_corner_2d.y
            print(f"Width: {bbox_width}")
            print(f"Height: {bbox_height}")
            print(f"Ratio: {bbox_width/bbox_height}")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Create absolute bounding box - finished ({exec_time:.4f} s)")

        # save the camera parameters to a csv file
        if STEPS["save_camera_params"]:

            print("Save camera parameters")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: save_camera_params - {current_date_time}\n")

            # get the camera params
            camera_params = ecp.extract_camera_params(chunk)

            # define safe path
            camera_params_path = os.path.join(project_fld, "camera_params.csv")

            # save to csv
            camera_params.to_csv(camera_params_path, index=False)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: save_camera_params - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Save camera parameters - finished ({exec_time:.4f} s)")

        if DISPLAY_STEPS["save_aoi"]:
            print("Save AOI")
            start_time = time.time()

            aoi_folder = os.path.join(data_fld, "aoi")
            if not os.path.exists(aoi_folder):
                os.makedirs(aoi_folder)

            # define path to save the aoi
            aoi_path = os.path.join(aoi_folder, "aoi.shp")

            # create shapely polygon from bounding box
            aoi = Polygon([(min_x, min_y), (min_x, max_y),
                           (max_x, max_y), (max_x, min_y), (min_x, min_y)])

            # Define the CRS using the EPSG code
            crs = CRS.from_epsg(epsg_code)

            # Create a GeoDataFrame with the polygon and CRS
            gdf = gpd.GeoDataFrame(index=[0], crs=crs.to_wkt(),
                                   geometry=[aoi])

            # Save the GeoDataFrame as a shapefile
            gdf.to_file(aoi_path)

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Save AOI - finished ({exec_time:.4f} s)")

        # build absolute mesh
        if STEPS["build_mesh_absolute"]:

            print("Build absolute mesh")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_mesh_absolute - {current_date_time}\n")

            # delete all existing meshes
            for model in chunk.models:
                chunk.remove(model)  # noqa
            doc.save()

            # arguments for building the mesh
            arguments = {
                'surface_type': Metashape.Arbitrary,
                'interpolation': interpolation,
                'replace_asset': True
            }

            # build mesh
            chunk.buildModel(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildModel_absolute",
                                   arguments, argument_fld)

            # save the project
            doc.save()

            print(" Export model to file")

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
                _save_command_args("exportModel_absolute",
                                   arguments, argument_fld)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_mesh_absolute - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute mesh - finished ({exec_time:.4f} s)")

        # build absolute point cloud
        if STEPS["build_pointcloud_absolute"]:

            print("Build absolute point cloud")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_pointcloud_absolute - {current_date_time}\n")

            # build parameters for the point cloud
            arguments = {
                'point_colors': True,
                'point_confidence': True,
                'replace_asset': True
            }

            print("Arguments:", arguments)

            # build dense cloud
            chunk.buildPointCloud(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildPointCloud_absolute",
                                   arguments, argument_fld)

            # save the project
            doc.save()

            print(" Export point cloud to file")

            # export parameters for the point cloud
            arguments = {
                'path': output_path_pc_abs,
                'crs': chunk.crs,
                'save_point_color': True,
                'save_point_confidence': True
            }

            print("Arguments:")
            print(arguments)

            # export the point cloud
            chunk.exportPointCloud(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("exportPointCloud_absolute",
                                   arguments, argument_fld)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_pointcloud_absolute - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute point cloud - finished ({exec_time:.4f} s)")

        # clean absolute point cloud
        if STEPS["clean_pointcloud_absolute"]:

            # only clean the pc if the original point cloud is still existing
            if os.path.exists(output_path_pc_abs) is False:

                # replace the point cloud path with the cleaned one
                output_path_pc_abs = output_path_pc_abs_c

                doc.save()

            else:
                print("Clean absolute point cloud")
                start_time = time.time()

                with open(steps_path, "a") as file:
                    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file.write(f"START: clean_pointcloud_absolute - {current_date_time}\n")

                # load the point cloud
                if point_cloud_abs is None:
                    point_cloud_abs = lpl.load_ply(output_path_pc_abs)

                # remove outliers
                point_cloud_abs_cleaned = ro.remove_outliers(point_cloud_abs)

                # convert numpy array to pandas dataframe
                cols = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                        'red', 'green', 'blue', 'class', 'confidence']
                point_cloud_abs_cleaned = pd.DataFrame(point_cloud_abs_cleaned,
                                                       columns=cols)

                # save the cleaned point cloud
                epc.export_pointcloud(point_cloud_abs_cleaned,
                                      output_path_pc_abs_c,
                                      overwrite=True)

                # delete all existing point clouds
                for pc in chunk.point_clouds:
                    chunk.remove(pc)  # noqa
                doc.save()

                # arguments for importing the point cloud
                arguments = {
                    'path': output_path_pc_abs_c,
                    'crs': chunk.crs,
                    'replace_asset': True
                }

                # import the point cloud into the chunk
                chunk.importPointCloud(**arguments)

                # rename the imported point cloud
                chunk.point_clouds[0].label = ''

                # delete the old point cloud from output
                os.remove(output_path_pc_abs)

                # replace the point cloud path with the cleaned one
                output_path_pc_abs = output_path_pc_abs_c

                doc.save()

                with open(steps_path, "a") as file:
                    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file.write(f"FINISHED: clean_pointcloud_absolute - {current_date_time}\n")

                finish_time = time.time()
                exec_time = finish_time - start_time
                print(f"Clean absolute point cloud - finished ({exec_time:.4f} s)")

        # set correct path for point cloud after cleaning
        if os.path.exists(output_path_pc_abs_c):
            output_path_pc_abs = output_path_pc_abs_c

        # build absolute DEM
        if STEPS["build_dem_absolute"]:

            print("Build absolute DEM")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_dem_absolute - {current_date_time}\n")

            # delete all existing DEMs
            for elev in chunk.elevations:
                chunk.remove(elev)  # noqa
            doc.save()

            # set build parameters for the DEM
            arguments = {
                'source_data': Metashape.DataSource.PointCloudData,
                'interpolation': interpolation,
                'projection': projection_abs,
                'resolution': resolution_abs,
                'replace_asset': True
            }

            # add region to build parameters
            if bounding_box_abs is not None:
                arguments['region'] = bounding_box_abs

            # build the DEM
            chunk.buildDem(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildDem_absolute", arguments,
                                   argument_fld)
    
            # save the project
            doc.save()

            # set export parameters for the DEM
            arguments = {
                'path': output_path_dem_abs,
                'source_data': Metashape.ElevationData,
                'image_format': Metashape.ImageFormatTIFF,
                'raster_transform': Metashape.RasterTransformNone,
                'nodata_value': no_data_value,
                'projection': projection_abs,
                'clip_to_boundary': True,
                'resolution_x': resolution_abs,
                'resolution_y': resolution_abs,
                'image_compression': compression
            }

            # add region to export parameters
            if bounding_box_abs is not None:
                arguments['region'] = bounding_box_abs

            print(arguments)

            # export the DEM
            chunk.exportRaster(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("exportDEM_absolute",
                                   arguments, argument_fld)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_dem_absolute - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute DEM - finished ({exec_time:.4f} s)")

        # build ortho-mosaic
        if STEPS["build_orthomosaic_absolute"]:

            print("Build absolute orthomosaic")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_orthomosaic_absolute - {current_date_time}\n")

            # delete all existing ortho-photos
            for ortho in chunk.orthomosaics:
                chunk.remove(ortho)  # noqa
            doc.save()

            # arguments for building the orthomosaic
            arguments = {
                'surface_data': Metashape.ModelData,
                'blending_mode': Metashape.MosaicBlending,
                'projection': projection_abs,
                'resolution': resolution_abs,
                'replace_asset': True
            }

            # add region to build parameters
            if bounding_box_abs is not None:
                arguments['region'] = bounding_box_abs

            # build the orthomosaic
            chunk.buildOrthomosaic(**arguments)

            # save the arguments of the command
            if save_commands:
                _save_command_args("buildOrthomosaic_absolute",
                                   arguments, argument_fld)

            # save the project
            doc.save()

            # set export parameters for the orthomosaic
            arguments = {
                'path': output_path_ortho_abs,
                'source_data': Metashape.OrthomosaicData,
                'image_format': Metashape.ImageFormatTIFF,
                'raster_transform': Metashape.RasterTransformNone,
                # 'nodata_value': no_data_value, only for DEMs
                'projection': projection_abs,
                'clip_to_boundary': True,
                'resolution_x': resolution_abs,
                'resolution_y': resolution_abs,
                'image_compression': compression
            }

            # add region to export parameters
            if bounding_box_abs is not None:
                arguments['region'] = bounding_box_abs

            # export the orthomosaic
            chunk.exportRaster(**arguments)

            if save_commands:
                _save_command_args("exportOrthomosaic_absolute",
                                   arguments, argument_fld)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_orthomosaic_absolute - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute orthomosaic - finished ({exec_time:.4f} s)")

        # align both images together (same size and same start point)
        if STEPS["build_dem_absolute"] and STEPS["build_orthomosaic_absolute"]:
            print("Align DEM and Ortho")

            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: align_dem_ortho - {current_date_time}\n")

            # align the images
            ai.align_images(output_path_ortho_abs, output_path_dem_abs,
                            output_path_ortho_abs, output_path_dem_abs)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: align_dem_ortho - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Align DEM and Ortho - finished ({exec_time:.4f} s)")

        # build confidence array
        if STEPS["build_confidence_absolute"]:
            print("Build absolute confidence array")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_confidence_absolute - {current_date_time}\n")

            # load the absolute dem
            if dem_abs is None:
                dem_abs, transform_abs = li.load_image(output_path_dem_abs,
                                                       return_transform=True)

            # load the point cloud
            if point_cloud_abs is None:
                point_cloud_abs = lpl.load_ply(output_path_pc_abs)

            # create absolute confidence array
            conf_arr_abs = cca.create_confidence_arr(dem_abs, point_cloud_abs,
                                                     transform_abs,
                                                     interpolate=True,
                                                     distance=10)

            # save the confidence array
            eti.export_tiff(conf_arr_abs, output_path_conf_abs,
                            transform=transform_abs, overwrite=True,
                            use_lzw=True)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_confidence_absolute - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute confidence array - finished ({exec_time:.4f} s)")

        # build the difference DEM
        if STEPS["build_difference_dem"]:

            print("Build absolute difference DEM")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_difference_dem - {current_date_time}\n")

            # load the absolute dem
            if dem_abs is None:
                dem_abs, transform_abs = li.load_image(output_path_dem_abs,
                                                       return_transform=True)

            # load the modern dem
            if dem_modern is None:
                bounds = cb.calc_bounds(transform_abs, dem_abs.shape)
                dem_modern = lr.load_rema(bounds,
                                          zoom_level=rema_level, auto_download=True)

            # get the bounds of the absolute dem
            bounds_abs = cb.calc_bounds(transform_abs, dem_abs.shape)

            # get relative dem and define the output path
            difference_dem_rel = cdd.create_difference_dem(dem_abs,
                                                           dem_modern,
                                                           bounds_abs,
                                                           "REMA10")
            # export the relative dem
            eti.export_tiff(difference_dem_rel, output_path_diff_rela,
                            transform=transform_abs, overwrite=True,
                            no_data=np.nan, use_lzw=True)

            # make relative dem absolute
            difference_dem_abs = np.abs(difference_dem_rel)

            # export the absolute dem
            eti.export_tiff(difference_dem_abs, output_path_diff_abs,
                            transform=transform_abs, overwrite=True,
                            no_data=np.nan, use_lzw=True)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_difference_dem - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute difference dem - finished ({exec_time:.4f} s)")

        # estimate the quality of the uncorrected DEM
        if STEPS["evaluate_dem"]:
            print("Evaluate dem")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: evaluate_dem - {current_date_time}\n")

            # load the absolute dem
            if dem_abs is None:
                dem_abs, transform_abs = li.load_image(output_path_dem_abs,
                                                       return_transform=True)

            if ortho_abs is None:
                ortho_abs = li.load_image(output_path_ortho_abs)

            # load the modern dem
            if dem_modern is None:
                bounds = cb.calc_bounds(transform_abs, dem_abs.shape)
                dem_modern, transform_rema = lr.load_rema(bounds,
                                                          zoom_level=rema_level,
                                                          return_transform=True,
                                                          auto_download=True)
            if ortho_modern is None:
                bounds = cb.calc_bounds(transform_abs, dem_abs.shape)
                ortho_modern = ls.load_satellite(bounds, return_transform=False)

            if rock_mask is None:
                if rock_mask_type == "REMA":
                    rock_mask = lrm.load_rock_mask(bounds, mask_resolution,
                                                   mask_buffer=gcp_mask_kernel_rock)
                elif rock_mask_type == "pixels":
                    tst = "old"
                    if tst == "modern":
                        rock_mask = np.zeros_like(ortho_modern[0, :, :])
                        condition = np.all((ortho_modern >= 40) & (ortho_modern <= 90), axis=0)  # noqa
                        rock_mask[condition] = 1
                    elif tst == "old":
                        rock_mask = np.zeros_like(ortho_abs)
                        rock_mask[ortho_abs < 50] = 1
                        rock_mask = rei.resize_image(rock_mask, ortho_modern.shape[-2:])
                else:
                    raise ValueError("Rock mask type not defined.")

            # resize the old dem to the modern dem shape
            dem_abs_resized = rei.resize_image(dem_abs, dem_modern.shape)

            # estimate the quality of the DEM
            quality_dict = edq.estimate_dem_quality(dem_abs_resized, dem_modern,
                                                    mask=rock_mask)

            # define the output path for the quality dict
            quality_path = os.path.join(output_fld,
                                        str(project_name + "_quality.json"))

            # export the quality dict to a json file
            with open(quality_path, 'w') as f:
                json.dump(quality_dict, f, indent=4)  # noqa

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: evaluate_dem - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Evaluate dem - finished ({exec_time:.4f} s)")

        # correct DEM by coregistration
        if STEPS["correct_dem"]:
            print("Correct DEM")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: correct_dem - {current_date_time}\n")

            # load the absolute dem
            if dem_abs is None:
                print("  Load absolute DEM")
                dem_abs, transform_abs = li.load_image(output_path_dem_abs,
                                                       return_transform=True)

            if ortho_abs is None:
                ortho_abs = li.load_image(output_path_ortho_abs)

            # load the modern dem
            print("  Load REMA")
            bounds = cb.calc_bounds(transform_abs, dem_abs.shape)
            dem_modern, transform_rema = lr.load_rema(bounds,
                                                      zoom_level=rema_level,
                                                      auto_download=True, return_transform=True)

            # load the modern ortho
            ortho_modern = ls.load_satellite(bounds, return_transform=False)

            old_shape = dem_abs.shape

            # resize the old dem to the modern dem shape
            dem_abs_resized = rei.resize_image(dem_abs, dem_modern.shape)

            # update the transform as well
            from affine import Affine
            transform_abs_resized = Affine(
                transform_rema.a,  # New x-scale
                transform_abs.b,  # Original x-skew
                transform_abs.c,  # Original x-offset
                transform_abs.d,  # Original y-skew
                transform_rema.e,  # New y-scale
                transform_abs.f  # Original y-offset
            )

            # create mask for the DEM
            if rock_mask_type == "REMA":
                rock_mask = lrm.load_rock_mask(bounds, mask_resolution, mask_buffer=gcp_mask_kernel_rock)
            elif rock_mask_type == "pixels":
                tst = "old"
                if tst == "modern":
                    rock_mask = np.zeros_like(ortho_modern[0, :, :])
                    condition = np.all((ortho_modern >= 40) & (ortho_modern <= 90), axis=0)  # noqa
                    rock_mask[condition] = 1
                elif tst == "old":
                    rock_mask = np.zeros_like(ortho_abs)
                    rock_mask[ortho_abs < 50] = 1
                    rock_mask = rei.resize_image(rock_mask, ortho_modern.shape[-2:])
            else:
                raise ValueError("Rock mask type not defined.")

            # correct the dem (with an adapted max_slope)
            dem_corrected, transform_corrected, new_max_slope = cd.correct_dem(
                dem_abs_resized, dem_modern,
                transform_abs_resized, transform_rema,
                modern_ortho=ortho_modern,
                mask=rock_mask,
                adapt_mask=True,
                max_slope=max_slope_begin)

            # resize back to the original shape
            dem_corrected = rei.resize_image(dem_corrected, old_shape)

            # adapt transform to reflect original pixel size
            transform_corrected = Affine(
                transform_abs.a,
                transform_corrected.b,
                transform_corrected.c,
                transform_corrected.d,
                transform_abs.e,
                transform_corrected.f
            )

            # adapt output path
            output_path_dem_corr = output_path_dem_abs.replace(".tif",
                                                               f"_{new_max_slope}.tif")

            # save the corrected dem
            eti.export_tiff(dem_corrected, output_path_dem_corr,
                            transform=transform_corrected, overwrite=True,
                            use_lzw=True, no_data=-9999)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: correct_dem - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Correct DEM - finished ({exec_time:.4f} s)")

        # build the difference DEM (corrected)
        if STEPS["build_difference_dem"] and STEPS["correct_dem"]:

            print("Build absolute difference DEM (corrected)")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: build_difference_dem_corrected - {current_date_time}\n")

            # load the corrected dem
            if dem_corrected is None or transform_corrected is None:
                dem_corrected, transform_corrected = li.load_image(output_path_dem_corr,
                                                                   return_transform=True)

            # get the bounds of the corrected dem
            bounds_corrected = cb.calc_bounds(transform_corrected, dem_corrected.shape)

            # load the modern dem again (for corrected values)
            dem_modern = lr.load_rema(bounds_corrected, zoom_level=rema_level,
                                      auto_download=True)

            # get relative dem and define the output path
            difference_dem_rela_c = cdd.create_difference_dem(dem_corrected,
                                                              dem_modern,
                                                              bounds_corrected,
                                                              "REMA10")

            # export the relative dem
            eti.export_tiff(difference_dem_rela_c, output_path_diff_rela_c,
                            transform=transform_corrected,
                            overwrite=True, no_data=np.nan,
                            use_lzw=True)

            # make relative dem absolute
            difference_dem_abs_c = np.abs(difference_dem_rela_c)

            # export the absolute dem
            eti.export_tiff(difference_dem_abs_c, output_path_diff_abs_c,
                            transform=transform_corrected,
                            overwrite=True, no_data=np.nan,
                            use_lzw=True)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: build_difference_dem_corrected - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Build absolute difference dem (corrected) - finished ({exec_time:.4f} s)")

        # estimate the quality of the corrected DEM
        if STEPS["evaluate_dem"] and STEPS["correct_dem"]:

            print("Evaluate corrected dem")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: evaluate_dem_corrected - {current_date_time}\n")

            # load the corrected dem
            if dem_corrected is None:
                dem_corrected, transform_corrected = li.load_image(output_path_dem_corr,
                                              return_transform=True)

            # calculate the bounds of the corrected dem
            bounds_corrected = cb.calc_bounds(transform_corrected,
                                              dem_corrected.shape)

            # load the modern dem
            dem_modern = lr.load_rema(bounds_corrected, zoom_level=rema_level,
                                      auto_download=True)

            if rock_mask_type == "REMA":
                rock_mask = lrm.load_rock_mask(bounds_corrected, mask_resolution,
                                               mask_buffer=gcp_mask_kernel_rock)
            elif rock_mask_type == "pixels":
                rock_mask = np.zeros_like(ortho_modern[0, :, :])
                condition = np.all((ortho_modern >= 40) & (ortho_modern <= 90), axis=0)  # noqa
                rock_mask[condition] = 1
            else:
                raise ValueError("Rock mask type not defined.")

            # resize the old dem to the modern dem shape
            dem_corrected_resized = rei.resize_image(dem_corrected, dem_modern.shape)

            # estimate the quality of the DEM
            quality_dict_corr = edq.estimate_dem_quality(dem_corrected_resized, dem_modern,
                                                    rock_mask)

            # define the output path for the quality dict
            quality_path_corr = os.path.join(output_fld,
                                        project_name + "_quality_corrected.json")

            # export the quality dict to a json file
            with open(quality_path_corr, 'w') as f:
                json.dump(quality_dict_corr, f, indent=4)  # noqa

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: evaluate_dem_corrected - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Evaluate corrected dem - finished ({exec_time:.4f} s)")

        if STEPS["evaluate_project"]:
            proj_qual_dict = gpq.get_project_quality(chunk)

        # create a report of the project
        if STEPS["create_report"]:
            print("Create report")
            start_time = time.time()

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: create_report - {current_date_time}\n")

            # create a report of the project
            report_path = os.path.join(project_fld, project_name + "_report.pdf")

            # create the report
            chunk.exportReport(report_path, title=project_name)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: create_report - {current_date_time}\n")

            finish_time = time.time()
            exec_time = finish_time - start_time
            print(f"Create report - finished ({exec_time:.4f} s)")

        # compress images in the 4 images folder
        if STEPS["compress_images"]:

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"START: compress_images - {current_date_time}\n")

            print(f"Compress images in '{img_folder}'")
            cotf.compress_tif_files(img_folder, "jpeg", 90)

            print(f"Compress images in 'enhanced_folder'")
            cotf.compress_tif_files(enhanced_folder, "jpeg", 90)

            mask_folder = os.path.join(data_fld, "masks_original")
            print(f"Compress images in '{mask_folder}'")
            cotf.compress_tif_files(mask_folder, "jpeg", 90)

            adapted_mask_folder = os.path.join(data_fld, "masks_adapted")
            print(f"Compress images in '{adapted_mask_folder}'")
            cotf.compress_tif_files(adapted_mask_folder, "jpeg", 90)

            with open(steps_path, "a") as file:
                current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"FINISHED: compress_images - {current_date_time}\n")

        # fix the output ortho
        if STEPS["fix_ortho"]:
            fo.fix_ortho(output_path_ortho_abs, 255)

        if STEPS["copy_to_external"]:
            pass


    except Exception as e:
        print("!!ERROR!!")
        print(e)
        traceback.print_exc()  # This prints the full stack trace

        # set error flag
        error_raised = True
        status_message = str(e)

        if DEBUG_MODE:
            raise e
    finally:

        # define path to save all parameters
        output_path_params = os.path.join(output_fld, "params.json")

        # create dict with the params
        params = {
            'project_name': project_name,
            'gcp_accuracy': gcp_accuracy,
            'azimuth': azimuth,
            'absolute_bounds': absolute_bounds,
            'epsg_code': epsg_code,
            'overwrite': overwrite,
            'resume': resume,
            'fixed_focal_length': fixed_focal_length,
            'use_rotations_only_for_tps': use_rotations_only_for_tps,
            'pixel_size': pixel_size,
            'resolution_rel': resolution_rel,
            'resolution_abs': resolution_abs,
            'matching_method': matching_method,
            'min_overlap': min_overlap,
            'step_range': step_range,
            'custom_matching': custom_matching,
            'min_tps': min_tps,
            'max_tps': max_tps,
            'min_tp_confidence': min_tp_confidence,
            'tp_type': tp_type,
            'tp_tolerance': tp_tolerance,
            'custom_markers': custom_markers,
            'zoom_level_dem': zoom_level_dem,
            'use_gcp_mask': use_gcp_mask,
            'mask_type': mask_type,
            'rock_mask_type': rock_mask_type,
            'mask_resolution': mask_resolution,
            'min_gcp_required': min_gcp_required,
            'min_gcp_confidence': min_gcp_confidence,
            'gcp_accuracy_px': gcp_accuracy_px,
            'min_markers': min_markers,
            'max_marker_error_px': max_marker_error_px,
            'max_marker_error_m': max_marker_error_m,
            'max_slope_begin': max_slope_begin,
            'max_slope_finish': max_slope_finish,
            'mask_buffer': mask_buffer,
            'no_data_value':no_data_value,
            'interpolate':interpolate
        }

        # special case for tp_type
        if params['tp_type'] is float:
            params['tp_type'] = "float"
        elif params['tp_type'] is int:
            params['tp_type'] = "int"

        # save the parameters of the project
        with open(output_path_params, 'w') as f:
            json.dump(params, f, indent=4, default=custom_serializer)  # noqa

        if error_raised is False:
            status = "finished"
            status_message = ""
        else:
            status = "error"

        # save to psql
        if save_to_psql:
            sstd.save_sfm_to_db(project_name, images_paths,
                                bounding_box_abs, status,
                                quality_dict, quality_dict_corr,
                                proj_qual_dict,
                                status_message=status_message, conn=conn)

        # close the text output
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
    if 'image_compression' in args:
        args['image_compression'] = "<TO BE DONE>"
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
    output_path_args = os.path.join(output_fld, f"{method}_args.json")

    # save the dict to a json file
    with open(output_path_args, 'w') as f:
        json.dump(args, f, indent=4)  # noqa


def custom_serializer(obj):
    """
    Custom serializer for JSON encoding.
    Converts non-serializable objects (e.g., sets, tuples) to serializable formats.
    """
    if isinstance(obj, set):  # Convert sets to lists
        return list(obj)
    elif hasattr(obj, '__dict__'):  # Convert objects with `__dict__` attribute to dicts
        return obj.__dict__
    elif isinstance(obj, type):  # Convert type objects to their string representation
        return obj.__name__
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":

    raise Exception("This script is not meant to be run as a standalone script. Please run 'start_agi'")