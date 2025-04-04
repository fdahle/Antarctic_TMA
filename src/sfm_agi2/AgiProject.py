"""the complete mega function to do SfM automatically"""

import json
import numpy as np
import Metashape
import os
import shutil
import yaml

from affine import Affine
from pathlib import Path

# base modules
import src.base.enhance_image as ei
import src.base.load_credentials as lc

# load modules
import src.load.load_image as li
import src.load.load_ply as lpl
import src.load.load_rema as lr
import src.load.load_rock_mask as lrm
import src.load.load_satellite as ls

# export modules
import src.export.export_tiff as eti

# snippets for prepare_images
import src.sfm_agi2.snippets.create_combinations as cc
import src.sfm_agi2.snippets.find_tie_points_for_sfm as ftp
import src.sfm_agi2.snippets.union_masks as um

# snippets for sfm
import src.sfm_agi2.snippets.create_confidence_arr as cca

# snippets for georef_project
import src.sfm_agi2.snippets.add_markers as am
import src.sfm_agi2.snippets.create_bundler as cb
import src.sfm_agi2.snippets.create_slope as cs
import src.sfm_agi2.snippets.georef_ortho as go
import src.sfm_agi2.snippets.find_gcps as fg
import src.sfm_agi2.snippets.filter_markers as fm

# snippets for correct_data
import src.sfm_agi2.snippets.correct_dem as cd

# snippets for evaluate_project
import src.sfm_agi2.snippets.estimate_dem_quality as edq

# constants
DEFAULT_IMAGE_FLD = "/data/ATM/data_1/aerial/TMA/downloaded"


class AgiProject:

    def __init__(self, project_name: str, base_fld: str,
                 agi_params: dict | None = None,
                 project_params: dict | None = None,
                 function_params: dict | None = None,
                 resume: bool = False, overwrite: bool = False,
                 debug: bool = False) -> None:
        """
        Initialize a new or existing Agisoft project.

        Args:
            project_name (str): Name of the project.
            base_fld (str): Base folder path where the project will be stored.
            resume (bool): Whether to resume an existing project.
            overwrite (bool): Whether to overwrite an existing project.
        """

        # basic project information
        self.project_name = project_name.lower().replace(" ", "_")
        self.project_fld = str(os.path.join(base_fld, self.project_name))
        self.project_psx_path = os.path.join(self.project_fld,
                                             f"{self.project_name}.psx")

        # initial check if combination of resume and overwrite is correct
        if resume is True and overwrite is True:
            raise ValueError("Cannot resume and overwrite at the same time. "
                             "Set one of them to False.")

        # initial safe check if project folder is already existing
        if os.path.exists(self.project_fld):

            # we overwrite the project folder
            if overwrite:
                shutil.rmtree(self.project_fld)
                os.makedirs(self.project_fld)
            # we resume the project
            elif resume:
                print(f"[INFO] Resuming project at {self.project_fld}")
            # File existing but no overwrite or resume
            else:
                raise FileExistsError(f"'{self.project_fld}' already exists. Use 'overwrite=True' or 'resume=True'.")
        else:
            os.makedirs(self.project_fld)

        # extended project information
        self.overwrite = overwrite
        self.resume = resume
        self.debug = debug

        # different subfolders
        self.data_fld = os.path.join(self.project_fld, "data")
        self.output_fld = os.path.join(self.project_fld, "output")
        self.log_fld = os.path.join(self.project_fld, "log")
        self.display_fld = os.path.join(self.project_fld, "display")

        ##
        # steps
        ##

        # path to the json file for steps
        self.json_steps_path = os.path.join(self.project_fld, "steps.json")

        # create a default step order
        self.step_order = [
            "prepare_images",
            "apply_rel_sfm",
            "georef_project",
            "apply_abs_sfm",
            "correct_data",
            "evaluate_project"
        ]

        # status flags to check progress
        if overwrite or resume is False:
            # create a new status flags file with all steps set to False
            self.status_flags = {step: False for step in self.step_order}

            # save the status flags to a json file
            self._update_status_flag_file()

        else:
            # load status flags from flags.json
            self.status_flags = self._load_status_flag_file()

        ##
        # params
        ##

        # load project_params (default if not provided)
        if project_params is None:
            path_default_project_params = ("/home/fdahle/Documents/GitHub/Antarctic_TMA/"
                                           "src/sfm_agi2/default_project_params.yaml")
            self.project_params = self._load_params(path_default_project_params)
        else:
            self.project_params = project_params

        # load agi_params (default if not provided)
        if agi_params is None:
            path_default_agi_params = ("/home/fdahle/Documents/GitHub/Antarctic_TMA/"
                                       "src/sfm_agi2/default_agi_params.yaml")
            self.agi_params = self._load_params(path_default_agi_params)
        else:
            self.agi_params = agi_params

        # load function params (default if not provided)
        if function_params is None:
            path_default_function_params = ("/home/fdahle/Documents/GitHub/Antarctic_TMA/"
                                             "src/sfm_agi2/default_function_params.yaml")
            self.function_params = self._load_params(path_default_function_params)
        else:
            self.function_params = function_params

        ##
        # availability flags for input data
        ##

        # path to the json file for steps
        self.json_data_availability_path = os.path.join(self.project_fld,
                                                        "data_availability.json")

        # create a default data availability file
        # create a default step order
        self.data_availability = {
            "images": False,
            "images_enhanced": False,
            "masks": False,
            "masks_adapted": False,
            "camera_positions": False,
            "camera_accuracies": False,
            "camera_rotations": False,
            "focal_lengths": False,
            "absolute_bounds": False,
        }

        # status flags to check data availability
        if overwrite or resume is False:

            # save the data availability to a json file
            self._update_data_availability_file()

        else:
            # load status flags from flags.json
            self.data_availability = self._load_data_availability_file()

        ###
        # intermediate data that we need at some point
        ###

        # the name of all images
        self.image_names = []

        # bounds of the project
        self.absolute_bounds = None  # absolute bounds of the project

        # DEMs
        self._dem_new = None  # e.g REMA
        self._dem_old_rel = None  # created with SFM in relative mode
        self._dem_old_abs = None  # created with SFM in absolute mode
        self._dem_old_corrected = None  # corrected dem_old_abs

        # transforms of DEMs
        self._dem_new_transform = None  # transform of the modern DEM
        self._dem_old_abs_transform = None  # transform of the absolute DEM
        self._dem_old_rel_transform = None  # transform of the relative DEM

        # masks of DEM
        self._dem_mask_new = None  # mask of the modern DEM
        self._dem_mask_old_abs = None  # mask of the old DEM (absolute one)
        self._dem_mask_old_rel = None  # mask of the old DEM (relative one)

        # Orthomosaics
        self._ortho_new = None  # e.g Sentinel-2
        self._ortho_old_rel = None  # created with SFM in relative mode
        self._ortho_old_abs = None  # created with SFM in absolute mode

        # transforms of Orthomosaics
        self._ortho_new_transform = None # transform of the modern ortho
        self._ortho_old_rel_transform = None # transform of the relative ortho
        self._ortho_old_abs_transform = None # transform of the absolute ortho

        # confidence arrays
        self._conf_arr_abs = None
        self._conf_arr_rel = None

        # Point clouds
        self._point_cloud_rel = None
        self._point_cloud_abs = None

        # diverse things
        self._rock_mask = None
        self._slope_new = None

        ##
        # Output paths
        ##
        self.pth_conf_arr_rel = os.path.join(self.output_fld,
                                                self.project_name + "_conf_arr_rel.tif")
        self.pth_conf_arr_abs = os.path.join(self.output_fld,
                                                self.project_name + "_conf_arr_abs.tif")
        self.pth_dem_rel = os.path.join(self.output_fld,
                                        self.project_name + "_dem_rel.tif")
        self.pth_dem_abs = os.path.join(self.output_fld,
                                        self.project_name + "_dem_abs.tif")
        self.pth_dem_corrected = os.path.join(self.output_fld,
                                              self.project_name + "_dem_corrected.tif")
        self.pth_ortho_rel = os.path.join(self.output_fld,
                                          self.project_name + "_ortho_rel.tif")
        self.pth_ortho_abs = os.path.join(self.output_fld,
                                          self.project_name + "_ortho_abs.tif")
        self.pth_pc_rel = os.path.join(self.output_fld,
                                       self.project_name + "_pc_rel.ply")
        self.pth_pc_abs = os.path.join(self.output_fld,
                                       self.project_name + "_pc_abs.ply")

        ##
        # Metashape initialization
        ##

        # get the license key and activate the license
        licence_key = lc.load_credentials("agisoft")['licence']
        Metashape.License().activate(licence_key)

        # enable use of gpu
        Metashape.app.gpu_mask = 1
        Metashape.app.cpu_enable = False

        # create compression object
        self.compression = Metashape.ImageCompression()
        self.compression.tiff_compression = Metashape.ImageCompression.TiffCompressionLZW
        self.compression.tiff_big = True

        # set up metashape
        self.doc = Metashape.Document(read_only=False)  # noqa
        if resume and os.path.exists(self.project_psx_path):
            print("[INFO] Opening existing project at {}.".format(self.project_psx_path))
            self.doc.open(self.project_psx_path, ignore_lock=True)
            self.chunk = self.doc.chunks[0]
        else:
            print("[INFO] Creating new project at {}.".format(self.project_psx_path))
            self.doc.save(self.project_psx_path)
            self.chunk = self.doc.addChunk()

        ##
        # project folder
        ##

        # create the folder structure
        for fld in [self.data_fld, self.output_fld, self.log_fld, self.display_fld]:
            os.makedirs(fld, exist_ok=True)

    @property
    def conf_arr_abs(self):
        """
        Lazy-loads the absolute confidence array.

        Returns:
            _conf_arr_abs np.ndarray: The confidence array generated from the
                absolute SFM step as a NumPy array.
        """


        if self._conf_arr_abs is None:
            self._conf_arr_abs = li.load_image(self.pth_conf_arr_abs)
        return self._conf_arr_abs

    @property
    def conf_arr_rel(self):
        """
        Lazy-loads the relative confidence array.

        Returns:
            _conf_arr_rel np.ndarray: The confidence array generated from the
                relative SFM step as a NumPy array.
        """

        if self._conf_arr_rel is None:
            self._conf_arr_rel = li.load_image(self.pth_conf_arr_rel)
        return self._conf_arr_rel

    @property
    def dem_new(self):

        if self.absolute_bounds is None:
            raise ValueError("Absolute bounds are required to load the modern DEM")

        if self._dem_new is None:
            self._dem_new, self._dem_new_transform = lr.load_rema(self.absolute_bounds,
                                         zoom_level=self.project_params["rema_zoom_level"],
                                         auto_download=True,
                                         return_transform=True)
        return self._dem_new, self._dem_new_transform

    @property
    def dem_old_abs(self) -> (np.ndarray, Affine):
        """
        Lazy-loads the absolute DEM and its associated affine transform
        if not already loaded.

        Returns:
            _dem_old_abs np.ndarray: The DEM generated from the absolute
                SFM step as a NumPy array.
            _dem_old_abs_transform: The affine transformation matrix
                associated with the DEM.
        """

        if self._dem_old_abs is None:
            self._dem_old_abs, self._dem_old_abs_transform = li.load_image(self.pth_dem_abs,
                                                                   return_transform=True)
        return self._dem_old_abs, self._dem_old_abs_transform

    @property
    def dem_old_rel(self) -> (np.ndarray, Affine):
        """
        Lazy-loads the relative DEM and its associated affine transform if not already loaded.

        Returns:
            _dem_old_rel np.ndarray: The DEM generated from the relative
                SFM step as a NumPy array.
            _dem_old_rel_transform: The affine transformation matrix
                associated with the DEM.
        """

        if self._dem_old_rel is None:
            self._dem_old_rel, self._dem_old_rel_transform = li.load_image(self.pth_dem_rel,
                                                                   return_transform=True)
        return self._dem_old_rel, self._dem_old_rel_transform

    @property
    def dem_mask_new(self, max_slope=90):

        if self._dem_mask_new is None:

            self._dem_mask_new = np.ones(self._dem_new.shape, dtype=bool)

            if self.project_params["dem_mask"]["use_rock"]:
                self._dem_mask_new[self.rock_mask == 0] = 0

            if self.project_params["dem_mask"]["use_slope"]:
                # load slope and apply to mask
                slope = self.slope_new
                self._dem_mask_new[slope > max_slope] = 0

        return self._dem_mask_new

    @property
    def dem_mask_old_abs(self):

        if self._dem_mask_old_abs is None:
            self._dem_mask_old_abs = np.ones(self.dem_old_abs.shape, dtype=bool)

            # filter based on confidence
            if self.project_params["dem_mask"]["use_confidence"]:
                self._dem_mask_old_abs[self.conf_arr_abs <
                                       self.function_params["dem_mask"]["min_confidence"]] = 0

        return self._dem_mask_old_abs

    @property
    def dem_mask_old_rel(self):

        if self._dem_mask_old_rel is None:
            self._dem_mask_old_rel = np.ones(self.dem_old_rel.shape, dtype=bool)

            # filter based on confidence
            if self.project_params["dem_mask"]["use_confidence"]:
                self._dem_mask_old_rel[self.conf_arr_rel <
                         self.function_params["dem_mask"]["min_confidence"]] = 0


        return self._dem_mask_old_rel

    @property
    def ortho_new(self):

        if self.absolute_bounds is None:
            raise ValueError("Absolute bounds are required to load the modern ortho")

        if self._ortho_new is None:
            self._ortho_new, self._ortho_new_transform = ls.load_satellite(self.absolute_bounds,
                                                                           return_transform=True)
        return self._ortho_new, self._ortho_new_transform

    @property
    def ortho_old_abs(self) -> (np.ndarray, Affine):
        """
        Lazy-loads the historic absolute orthoimage and its associated
        affine transform if not already loaded.

        Returns:
            _ortho_old np.ndarray: The historic orthoimage as a NumPy array.
            _ortho_old_transform: The affine transformation matrix
                associated with the orthoimage.
        """

        if self._ortho_old_abs is None:
            self._ortho_old_abs, self._ortho_old_abs_transform = li.load_image(self.pth_ortho_abs,
                                                                               return_transform=True)
        return self._ortho_old_abs, self._ortho_old_abs_transform

    @property
    def ortho_old_rel(self):
        """
        Lazy-loads the historic relative orthoimage and its associated
        affine transform if not already loaded.

        Returns:
            _ortho_rel np.ndarray: The historic orthoimage as a NumPy array.
            _ortho_rel_transform: The affine transformation matrix
                associated with the orthoimage.
        """

        if self._ortho_old_rel is None:
            self._ortho_old_rel, self._ortho_old_rel_transform = li.load_image(self.pth_ortho_rel,
                                                                               return_transform=True)
        return self._ortho_old_rel, self._ortho_old_rel_transform

    @property
    def point_cloud_abs(self) -> np.ndarray:
        """
        Lazy-loads the absolute point cloud (.ply) if not already loaded.

        Returns:
            _point_cloud_abs (np.ndarray): The absolute point cloud generated
                after absolute SFM.
        """

        if self._point_cloud_abs is None:
            self._point_cloud_abs = lpl.load_ply(self.pth_pc_abs)
        return self._point_cloud_abs

    @property
    def point_cloud_rel(self):
        """
        Lazy-loads the relative point cloud (.ply) if not already loaded.

        Returns:
            _point_cloud_rel (np.ndarray): The relative point cloud generated
                after relative SFM.
        """


        if self._point_cloud_rel is None:
            self._point_cloud_rel = lpl.load_ply(self.pth_pc_rel)
        return self._point_cloud_rel

    @property
    def rock_mask(self):
        if self._rock_mask is None:
            # load rock mask and apply to mask
            self._rock_mask = lrm.load_rock_mask(self.absolute_bounds,
                                           self.project_params["rema_zoom_level"],
                                           self.project_params["rema_buffer"])
        return self._rock_mask

    @property
    def slope_new(self) -> np.ndarray:
        """
        Lazy-loads and create the slope map of the new (=modern) DEM.

        Returns:
            _slope_new (np.ndarray): The slope of the new/modern DEM in degrees.
        """

        if self._slope_new is None:
            # get dem and transform
            d_new, t_new = self.dem_new
            self._slope_new = cs.create_slope(d_new, t_new,
                                              self.project_params["epsg_code"],
                                              self.function_params["no_data"])
        return self._slope_new

    def set_input_images(self, image_names: list[str],
                         src_folder: str | None = None,
                         extension: str = ".tif") -> None:
        """
        Set and copy input images into the project folder.

        This function ensures all input images have the specified file extension,
        checks their existence in the source folder, and copies them to the
        project's data directory.

        Args:
            image_names (list[str]): List of image names (with or without extension).
            src_folder (str | None): Path to the folder containing the source images.
                Defaults to `DEFAULT_IMAGE_FLD` if not provided.
            extension (str): File extension to enforce (e.g., '.tif', '.tiff'). Default is '.tif'.

        Raises:
            ValueError: If fewer than 3 images are provided.
            FileNotFoundError: If a source image does not exist in the source folder.
        """

        print("[INFO] Setting {} input images.".format(len(image_names)))

        if len(image_names) < 3:
            raise ValueError("At least 3 images are required for SFM.")

        if src_folder is None:
            src_folder = DEFAULT_IMAGE_FLD

        # save the images
        self.image_names = image_names

        # create subfolder for images
        image_folder = os.path.join(self.data_fld, "images")
        os.makedirs(image_folder, exist_ok=True)

        # copy images to the data folder
        for image_name in self.image_names:

            # check if the image name is correct
            if image_name.endswith("."+ extension) is False:
                image_name = image_name + "." + extension

            # set the correct path
            source_path = str(os.path.join(src_folder, image_name))
            dest_path = str(os.path.join(image_folder, image_name))

            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source image not found: {source_path}")

            # copy the image
            shutil.copy(source_path, dest_path)

        self._update_data_availability_file("images")

    def set_optional_data(self,
                          absolute_bounds=None,
                          ortho_new=None, ortho_new_transform=None,
                          dem_new=None, dem_new_transform=None,
                          masks=None, mask_src_folder: str | None = None,
                          camera_positions=None, camera_footprints=None,
                          camera_accuracies=None,
                          focal_lengths = None):

        # define paths for the optional data
        mask_folder = os.path.join(self.data_fld, "masks")

        # before optional data can be set, check if images are set
        if len(self.image_names) < 3:
            raise ValueError("Images must be set before setting input data.")

        if absolute_bounds is not None:
            self.absolute_bounds = absolute_bounds

        if masks is not None:
            if mask_src_folder is None:
                raise ValueError("Mask source folder must be set if masks "
                                 "are provided.")

            if len(masks) != len(self.image_names):
                raise ValueError("Number of masks must match number of images.")

            # copy masks to the data folder
            for mask_name in masks:
                source_path = str(os.path.join(mask_src_folder, mask_name))
                dest_path = str(os.path.join(mask_folder, mask_name))
                shutil.copy(source_path, dest_path)

            self._update_data_availability_file("masks")

        if camera_positions is not None:

            if len(camera_positions) != len(self.image_names):
                raise ValueError("Number of camera positions must match "
                                 "number of images.")
            self._update_data_availability_file("camera_positions")

        if camera_accuracies is not None:
            if len(camera_accuracies) != len(self.image_names):
                raise ValueError("Number of camera accuracies must match "
                                 "number of images.")
            self._update_data_availability_file("camera_accuracies")

        if focal_lengths is not None:
            if len(focal_lengths) != len(self.image_names):
                raise ValueError("Number of focal lengths must match "
                                 "number of images.")
            self._update_data_availability_file("focal_lengths")

    def run_project(self):
        """run project with all steps"""

        self.prepare_images()
        self.apply_rel_sfm()
        self.georef_project()
        self.apply_abs_sfm()
        self.correct_data()
        self.evaluate_project()

    def prepare_images(self):
        # check overwrite (no need checking for previous steps, as first step)
        if self.status_flags["prepare_images"]:
            if self.overwrite is False:
                print("[INFO] Images already prepared. Skipping preparation.")
                return

        # check image availability
        if self.data_availability["images"] is False:
            raise RuntimeError("No images available. Please set images first.")

        # set path to the possible folder
        image_folder = os.path.join(self.data_fld, "images")
        enhanced_folder = os.path.join(self.data_fld, "enhanced_images")
        orig_mask_folder = os.path.join(self.data_fld, "masks")
        adapted_mask_folder = os.path.join(self.data_fld, "masks_adapted")

        # get all tif files in the image folder
        lst_images = [p.resolve() for p in Path(image_folder).glob('*.tif')]

        # add the images to the chunk
        self.chunk.addPhotos(lst_images)

        # set images to film cameras (for finding fiducials)
        for camera in self.chunk.cameras:

            # skip cameras that are not enabled
            if camera.enabled is False:
                continue

            # set camera to film camera
            camera.sensor.film_camera = True
            camera.sensor.fixed_calibration = True

        # find fiducials
        arguments = self.agi_params["detectFiducials"]
        self.chunk.detectFiducials(**arguments)

        # create folder for adapted masks
        os.makedirs(adapted_mask_folder, exist_ok=True)

        # union masks to create adapted masks
        if self.data_availability["masks"]:

            # union masks
            um.union_masks(self.chunk, orig_mask_folder, adapted_mask_folder)
            self._update_data_availability_file("adapted_masks")

            mask_folder = adapted_mask_folder
        else:
            mask_folder = orig_mask_folder


        # enhance the images
        if self.project_params["enhance_images"]:

            # create folder for enhanced images
            os.makedirs(enhanced_folder, exist_ok=True)

            # iterate all tif files in image folder
            for img_pth in lst_images:

                # get the base name of the image
                base_name = os.path.basename(img_pth)

                # load the image
                image = li.load_image(img_pth)

                # load the mask
                mask_path = os.path.join(mask_folder, base_name)
                mask = li.load_image(mask_path)

                # enhance the image
                enhanced_image = ei.enhance_image(image, mask)

                # save the enhanced image
                enhanced_path = os.path.join(enhanced_folder, base_name)
                eti.export_tiff(enhanced_image, enhanced_path, use_lzw=True)

            # set flag for enhanced images
            self._update_data_availability_file("enhanced_images")


        # check if images must be rotated (for tp matching)

        # get the combinations for matching
        arguments = self.project_params["combinations_params"]
        combinations = cc.create_combinations(**arguments)


        # match the images
        if self.project_params["custom_matching"]:

            # find tie points with our own custom matching
            arguments = self.project_params["custom_matching_params"]
            tp_dict, conf_dict = ftp.find_tie_points_for_sfm(image_folder, combinations,
                                                             mask_folder=mask_folder,
                                                             rotation_dict=rotation_dict,
                                                             **arguments)

            # create the bundler file from the tie points
            path_bundler_file = cb.create_bundler(tp_dict, conf_dict)

            # import the bundler file
            self.chunk.importCameras(path_bundler_file,
                                     format=Metashape.CamerasFormatBundler)

        else:

            arguments = self.agi_params["matchPhotos"]
            self.chunk.matchPhotos(**arguments)
            self.doc.save()

        # set status flag to true
        self._update_status_flag_file("prepare_images")

    def apply_rel_sfm(self):
        # check previous steps and also overwrite
        self._check_previous_steps("apply_rel_sfm")
        if self.status_flags["apply_rel_sfm"]:
            if self.overwrite is False:
                print("[INFO] Relative SFM already applied. Skipping.")
                return

        # align cameras
        arguments = self.agi_params["alignCameras_relative"]
        # we need to reset the alignment if custom matching was used
        if self.project_params["custom_matching"]:
            arguments['reset_alignment'] = True
        self.chunk.alignCameras(**arguments)
        self.doc.save()

        # check if cameras are aligned
        num_cameras = len(self.chunk.cameras)
        num_aligned = 0
        for camera in self.chunk.cameras:
            if camera.enabled and camera.transform:
                num_aligned += 1

        if num_aligned < 3:
            raise RuntimeError("Not enough cameras aligned. "
                               "Please check the input images.")
        else:
            print(f"[INFO] {num_aligned}/{num_cameras} cameras aligned.")

        # build depth maps
        arguments = self.agi_params["buildDepthMaps_relative"]
        self.chunk.buildDepthMaps(**arguments)
        self.doc.save()

        # build mesh
        arguments = self.agi_params["buildModel_relative"]
        self.chunk.buildModel(**arguments)
        self.doc.save()

        # build point cloud
        arguments = self.agi_params["buildPointCloud_relative"]
        self.chunk.buildPointCloud(**arguments)
        self.doc.save()

        # build DEM
        arguments = self.agi_params["buildDem_relative"]
        self.chunk.buildDem(**arguments)
        self.doc.save()

        # build ortho
        arguments = self.agi_params["buildOrthomosaic_relative"]
        self.chunk.buildOrthomosaic(**arguments)
        self.doc.save()

        # build confidence arr
        d_rel, t_rel = self.dem_old_rel
        arguments = self.function_params["create_confidence_arr"]
        arguments["no_data_val"] = self.function_params["no_data"]
        conf_arr_rel = cca.create_confidence_arr(d_rel, self.point_cloud_rel, t_rel,
                                  **arguments)

        eti.export_tiff(conf_arr_rel, self.pth_conf_arr_rel,
                        transform=t_rel, use_lzw=True)

        # set status flag to true
        self._update_status_flag_file("apply_rel_sfm")

    def georef_project(self):
        # check previous steps and also overwrite
        self._check_previous_steps("georef_project")
        if self.status_flags["georef_project"]:
            if self.overwrite is False:
                print("[INFO] Geo-referencing already applied. Skipping.")
                return

        # stop if we are missing absolute bounds or camera footprints
        if self.data_availability["absolute_bounds"] is False or \
            self.data_availability["camera_footprints"] is False:
            raise ValueError("Missing absolute bounds or camera footprints "
                             "for geo-referencing.")

        # get orthos
        ortho_old_rel, _ = self.ortho_old_rel
        ortho_new, transform_new = self.ortho_new

        # georeference the relative ortho
        arguments = self.function_params["georef_ortho_params"]
        arguments["tp_type"] = self.project_params["tp_type"]
        tpl = go.georef_ortho(relative_ortho=ortho_old_rel,
                              absolute_ortho=ortho_new,
                              absolute_transform=transform_new,
                              **arguments)

        # extract values from return object
        transform_georef = tpl[0]
        bounds_georef = tpl[1]

        # get dems
        dem_old_rel, _ = self.dem_old_rel
        dem_new, transform_new = self.dem_new

        # get dem masks

        # find gcps
        gcps = fg.find_gcps(ortho_old_rel, ortho_new,
                            self.dem_old_rel, self.dem_new,
                            self.dem_mask_old_rel, self.dem_mask_new
                            )

        # set expected accuracy of the markers in px
        self.chunk.marker_projection_accuracy = gcp_accuracy_px

        # add gcps as markers
        arguments = self.function_params["add_markers_params"]
        am.add_markers(self.chunk, gcps,
                       self.project_params["epsg_code"],
                       **arguments)

        # set complete project to absolute coordinates
        epsg_code = self.project_params["epsg_code"]
        self.chunk.crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa
        chunk.camera_crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa
        self.doc.save()

        # filter markers
        if self.project_params["filter_markers"]:
            arguments = self.function_params["filter_markers_params"]
            fm.filter_markers(self.chunk, **arguments)
        else:
            # even without filtering, we need to optimize cameras at least once
            self.chunk.optimizeCameras()
            self.chunk.updateTransform()

        self.doc.save()

        # set status flag to true
        self._update_status_flag_file("georef_project")

    def apply_abs_sfm(self):

        # check previous steps and also overwrite
        self._check_previous_steps("apply_abs_sfm")
        if self.status_flags["apply_abs_sfm"]:
            if self.overwrite is False:
                print("[INFO] Absolute SFM already applied. Skipping.")
                return

        # build depth maps
        arguments = self.agi_params["buildDepthMaps_absolute"]
        self.chunk.buildDepthMaps(**arguments)
        self.doc.save()

        # build mesh
        arguments = self.agi_params["buildModel_absolute"]
        self.chunk.buildModel(**arguments)
        self.doc.save()

        # build point cloud
        arguments = self.agi_params["buildPointCloud_absolute"]
        self.chunk.buildPointCloud(**arguments)
        self.doc.save()

        # export point cloud
        arguments = self.agi_params["exportPointCloud_absolute"]
        arguments['path'] = self.pth_pc_abs
        self.chunk.exportPointCloud(**arguments)

        # build DEM
        arguments = self.agi_params["buildDem_absolute"]
        self.chunk.buildDem(**arguments)
        self.doc.save()

        # export DEM
        arguments = self.agi_params["exportDem_absolute"]
        arguments['path'] = self.pth_dem_abs
        self.chunk.exportRaster(**arguments)

        # build ortho
        arguments = self.agi_params["buildOrthomosaic_absolute"]
        self.chunk.buildOrthomosaic(**arguments)
        self.doc.save()

        # adapt arguments for export
        arguments = self.agi_params["exportOrthomosaic_absolute"]
        arguments['path'] = self.pth_ortho_abs

        if self.project_params["compress_output"]:
            arguments['compression'] = self.compression


        # export ortho
        self.chunk.exportRaster(**arguments)

        # build confidence arr
        d_abs, t_abs = self.dem_old_abs
        arguments = self.function_params["create_confidence_arr"]
        arguments["no_data_val"] = self.function_params["no_data"]
        conf_arr_abs = cca.create_confidence_arr(d_abs, self.point_cloud_abs, t_abs,
                                  **arguments)

        # export confidence arr
        eti.export_tiff(conf_arr_abs, self.pth_conf_arr_abs,
                        transform=t_abs, use_lzw=True)

        # set status flag to true
        self._update_status_flag_file("apply_abs_sfm")

    def correct_data(self):

        # check previous steps and also overwrite
        self._check_previous_steps("correct_data")
        if self.status_flags["correct_data"]:
            if self.overwrite is False:
                print("[INFO] Data already corrected. Skipping.")
                return

        # correct dem
        tpl = cd.correct_dem(self.dem_old_abs, self.dem_new, self.slope_new)

        self._dem_old_corrected = tpl[0]
        self._dem_old_corrected_transform = tpl[1]

        # export corrected DEM
        eti.export_tiff(self._dem_old_corrected, self.pth_dem_corrected,
                        self._dem_old_corrected_transform,
                        use_lzw=True)

        # set status flag to true
        self._update_status_flag_file("correct_data")

    def evaluate_project(self):

        # check previous steps and also overwrite
        self._check_previous_steps("evaluate_project")
        if self.status_flags["evaluate_project"]:
            if self.overwrite is False:
                print("[INFO] Project already evaluated. Skipping.")
                return

        edq.estimate_dem_quality(self.dem_old_abs, self.dem_new)

        # set status flag to true
        self._update_status_flag_file("evaluate_project")

    def _check_previous_steps(self, current_step: str) -> None:
        """
        Raise an error if any step before `current_step` has not been completed.

        Args:
            current_step (str): The step to check against.

        Raises:
            ValueError: If the current step is not in the step order.
            RuntimeError: If any preceding step has not been completed.
        """
        if current_step not in self.step_order:
            raise ValueError(f"Unknown step: {current_step}")

        current_index = self.step_order.index(current_step)
        for step in self.step_order[:current_index]:
            if not self.status_flags.get(step, False):
                raise RuntimeError(f"Cannot run '{current_step}' because '{step}' is not completed.")

    def _load_status_flag_file(self) -> dict[str, bool]:
        """
        Load status flags from disk if available, otherwise initialize
        dict with all steps set to False.

        Returns:
            dict[str, bool]: Dictionary of processing step flags.
        """
        if not os.path.exists(self.json_steps_path):
            print("[WARNING] Status file not found, initializing default flags.")
            return {step: False for step in self.step_order}

        with open(self.json_steps_path, "r") as f:
            return json.load(f)

    def _update_status_flag_file(self, step: str | None = None) -> None:
        """
        Mark the given step as completed, reset all subsequent steps,
        and update the status file.

        Args:
            step (str): The step to mark as completed.

        Raises:
            ValueError: If the step is not recognized.
        """

        if step is not None:
            if step not in self.status_flags:
                raise ValueError(f"Unknown step: {step}")

            # Mark the current step as done
            self.status_flags[step] = True

            # Reset all following steps
            if step in self.step_order:
                current_index = self.step_order.index(step)
                for later_step in self.step_order[current_index + 1:]:
                    self.status_flags[later_step] = False
            else:
                print(f"[WARNING] Step '{step}' not found in step order — skipping downstream reset.")

        with open(self.json_steps_path, "w") as f:
            json.dump(self.status_flags, f, indent=2)  # noqa

    def _load_data_availability_file(self) -> dict[str, bool]:
        pass

    def _update_data_availability_file(self,
                                       data_type: str | None = None) -> None:

        if data_type is not None:
            if data_type not in self.data_availability:
                raise ValueError(f"Unknown data type: {data_type}")

            # Mark the current step as done
            self.data_availability[data_type] = True

        with open(self.json_data_availability_path, "w") as f:
            json.dump(self.data_availability, f, indent=2)

    def _load_params(self, yaml_path: str) -> dict:

        def __resolve(value):
            if isinstance(value, str) and value.startswith("Metashape."):
                return getattr(Metashape, value.split(".")[1])

        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        return __resolve(raw)


if __name__ == "__main__":

    # create a new AgiProject
    _project_name = "test"
    _base_fld = "/data/ATM/data_1/sfm/agi_projects2"

    input_images = ["CA215332V0419", "CA215332V0420", "CA215332V0421",
                    "CA215332V0422", "CA215332V0423", "CA215332V0424",
                    "CA215332V0425"]

    for img in input_images:
        li.load_image(img)

    ap = AgiProject(_project_name, _base_fld, overwrite=True)
    ap.set_input_images(input_images)