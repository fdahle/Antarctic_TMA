"""the complete mega function to do SfM automatically"""
import sys
import os
from contextlib import contextmanager, redirect_stdout

from numpy.core.numeric import isscalar

# ignore some warnings
os.environ['KMP_WARNINGS'] = '0'
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

import gc
import hashlib
import json
import math
import Metashape
import numpy as np
import shutil
import yaml

from pathlib import Path
from rasterio.transform import Affine
from scipy.ndimage import binary_dilation
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from tqdm import tqdm
from time import sleep

# base modules
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
import src.sfm_agi2.snippets.enhance_images as ei
import src.sfm_agi2.snippets.find_tie_points_for_sfm as ftp
import src.sfm_agi2.snippets.union_masks as um

# snippets for sfm
import src.sfm_agi2.snippets.create_confidence_arr as cca

# snippets for georef_project
import src.sfm_agi2.snippets.adapt_bounds as ab
import src.sfm_agi2.snippets.add_markers as am
import src.sfm_agi2.snippets.create_bundler as cb
import src.sfm_agi2.snippets.create_slope as cs
import src.sfm_agi2.snippets.crop_output as co
import src.sfm_agi2.snippets.georef_ortho as go
import src.sfm_agi2.snippets.find_gcps as fg
import src.sfm_agi2.snippets.find_gcps2 as fg2
import src.sfm_agi2.snippets.filter_markers as fm

# snippets for correct_data
import src.sfm_agi2.snippets.correct_dem as cd

# snippets for evaluate_project
import src.sfm_agi2.snippets.estimate_dem_quality as edq

# other imports
from src.sfm_agi2.SfMError import SfMError

# constants
DEFAULT_IMAGE_FLD = "/data/ATM/data_1/aerial/TMA/downloaded"

@contextmanager
def redirect_output_to_file(log_path, debug: bool = False):
    """
    Redirects both Python-level and C-level (file descriptor)
    output to a log file.
    """
    if debug:
        class Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()

            def flush(self):
                for s in self.streams:
                    s.flush()

        with open(log_path, 'a') as logfile:
            tee = Tee(sys.stdout, logfile)
            with redirect_stdout(tee):
                yield
    else:
        # Save original file descriptors for stdout (1)
        original_stdout_fd = os.dup(1)

        # Open the log file for writing/appending
        log_fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)

        # Redirect file descriptors 1 (stdout)
        os.dup2(log_fd, 1)

        try:
            yield
        finally:
            # Restore the original file descriptors
            os.dup2(original_stdout_fd, 1)
            os.close(original_stdout_fd)
            os.close(log_fd)


class AgiProject:
    """
    The AgiProject class is used to create and manage an Agisoft project.
    """

    def __init__(self, project_name: str, base_fld: str,
                 agi_params: dict | None = None,
                 project_params: dict | None = None,
                 function_params: dict | None = None,
                 cache_params: dict | None = None,
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
                print(f"[INFO] Remove existing project folder at {self.project_fld}")
                shutil.rmtree(self.project_fld)
                os.makedirs(self.project_fld)
            # we resume the project
            elif resume:
                print(f"[INFO] Resuming project at {self.project_fld}")
            # File existing but no overwrite or resume
            else:
                raise FileExistsError(f"'{self.project_fld}' already exists. Use 'overwrite=True' or 'resume=True'.")
        else:

            # change status flag as folder is not existing
            overwrite = True
            resume = False

            # create folder
            os.makedirs(self.project_fld)

        # extended project information
        self.overwrite = overwrite
        self.resume = resume
        self.debug = debug

        # different subfolders
        self.data_fld = os.path.join(self.project_fld, "data")
        self.output_fld = os.path.join(self.project_fld, "output")
        self.log_fld = os.path.join(self.project_fld, "log")
        self.params_fld = os.path.join(self.project_fld, "params")
        self.debug_fld = os.path.join(self.project_fld, "debug")

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
        # project folders
        ##

        # create the folder structure
        for fld in [self.data_fld, self.output_fld,
                    self.log_fld, self.params_fld,
                    self.debug_fld]:
            os.makedirs(fld, exist_ok=True)

        ##
        # params
        ##

        # load project_params (default if not provided)
        if project_params is None:
            default_pth = ("/home/fdahle/Documents/GitHub/Antarctic_TMA/"
                           "src/sfm_agi2/default_project_params.yaml")
            new_pth = os.path.join(self.params_fld, "project_params.yaml")
            shutil.copy(default_pth, new_pth)
            self.project_params = self._load_params(new_pth)
        else:
            self.project_params = project_params

        # load agi_params (default if not provided)
        if agi_params is None:
            default_pth = ("/home/fdahle/Documents/GitHub/Antarctic_TMA/"
                           "src/sfm_agi2/default_agi_params.yaml")
            new_pth = os.path.join(self.params_fld, "agi_params.yaml")
            shutil.copy(default_pth, new_pth)
            self.agi_params = self._load_params(new_pth)
        else:
            self.agi_params = agi_params

        # load function params (default if not provided)
        if function_params is None:
            default_pth = ("/home/fdahle/Documents/GitHub/Antarctic_TMA/"
                           "src/sfm_agi2/default_function_params.yaml")
            new_pth = os.path.join(self.params_fld, "function_params.yaml")
            shutil.copy(default_pth, new_pth)
            self.function_params = self._load_params(new_pth)
        else:
            self.function_params = function_params

        # load cache params (default if not provided)
        if cache_params is None:
            default_pth = ("/home/fdahle/Documents/GitHub/Antarctic_TMA/"
                           "src/sfm_agi2/default_cache_params.yaml")
            new_pth = os.path.join(self.params_fld, "cache_params.yaml")
            shutil.copy(default_pth, new_pth)
            self.cache_params = self._load_params(new_pth)
        else:
            self.cache_params = cache_params

        ##
        # availability flags for input data
        ##

        # path to the json file for steps
        self.json_data_availability_path = os.path.join(self.project_fld,
                                                        "data_availability.json")

        # status flags to check data availability
        if overwrite or resume is False:

            # create a default data availability file
            self.data_availability = {
                "images": False,
                "images_enhanced": False,
                "masks": False,
                "masks_adapted": False,
                "camera_accuracies": False,
                "camera_footprints": False,
                "camera_positions": False,
                "camera_rotations": False,
                "focal_lengths": False,
                "absolute_bounds": False,
                "custom_ortho": False,
                "custom_dem": False
            }

            # save the data availability to a json file
            self._update_data_availability_file()
        else:
            # load data availability from json file
            self.data_availability = self._load_data_availability_file()

        ##
        # bounds
        ##
        self.relative_bounds = None
        self.absolute_bounds = None

        ##
        # optional data
        #

        self.camera_accuracies = None  # dict with camera accuracies
        self.camera_positions = None
        self.camera_footprints = None
        self.camera_rotations = None
        self.focal_lengths = None

        ###
        # intermediate data that we need at some point
        ###

        # the name of all images
        self.image_names = []

        # DEMs
        self._dem_new = None  # e.g REMA
        self._dem_old_rel = None  # created with SFM in relative mode
        self._dem_old_abs = None  # created with SFM in absolute mode
        self._dem_old_corrected = None  # corrected dem_old_abs

        # transforms of DEMs
        self._dem_new_transform = None  # transform of the modern DEM
        self._dem_old_abs_transform = None  # transform of the absolute DEM
        self._dem_old_rel_transform = None  # transform of the relative DEM
        self._dem_old_corrected_transform = None  # transform of the corrected DEM

        # masks of DEM
        self._dem_mask_new = None  # mask of the modern DEM
        self._dem_mask_old_abs = None  # mask of the old DEM (absolute one)
        self._dem_mask_old_rel = None  # mask of the old DEM (relative one)

        # attribute related to the masks of the dem
        self.min_slope = self.project_params["dem_mask"]["slope_min"]
        self.max_slope = self.project_params["dem_mask"]["slope_max"]
        self.active_slope = "gcps"  # can be "gcps" or "eval"
        self.gcp_slope = self.min_slope
        self.eval_slope = self.min_slope

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

        self.pth_log = os.path.join(self.log_fld, "output.log")

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
            print("[INFO] Create new project at {}.".format(self.project_psx_path))

            # !IMPORTANT: doc must be saved before adding a chunk, otherwise
            # the chunk will not be saved
            self.doc.save(self.project_psx_path)
            self.chunk = self.doc.addChunk()

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
    def dem_new(self) -> (np.ndarray, Affine):
        """
        Lazy-loads the modern DEM and its associated affine transform from
        REMA if not already loaded. It will be downloaded automatically
        if not available locally.

        Returns:
            _dem_new np.ndarray: The modern DEM.
            _dem_new_transform: The affine transformation matrix
                associated with the DEM.
        Raises:
            ValueError: If `absolute_bounds` is not set.
        """

        if self.absolute_bounds is None:
            raise ValueError("Absolute bounds are required to load the modern DEM")

        if self._dem_new is None:

            # get rema zoom level
            zoom_lvl = self.project_params["rema"]["zoom_level"]

            # check if we can use a cached version
            if self.cache_params["use_rema"] and self.cache_params["use_cache"]:

                # define path to cached file
                file_name = f"rema_{self.absolute_bounds[0]}_{self.absolute_bounds[1]}_" \
                    f"{self.absolute_bounds[2]}_{self.absolute_bounds[3]}_{zoom_lvl}m.tif"
                pth_cached_file= os.path.join(self.data_fld, "rema", file_name)

                # check if there is a cached version
                if os.path.isfile(pth_cached_file):
                    print(f"[INFO] Load cached modern elevation data")
                    self._dem_new, self._dem_new_transform = li.load_image(pth_cached_file,
                                                                         return_transform=True)

            # need to load the rema data directly if dem_new is still none
            if self._dem_new is None:

                print(f"[INFO] Load modern elevation data from REMA ({zoom_lvl}m) with", self.absolute_bounds)
                self._dem_new, self._dem_new_transform = lr.load_rema(self.absolute_bounds,
                                             zoom_level=zoom_lvl,
                                             auto_download=True,
                                             return_transform=True)

            # save the dem to the cache
            if self.cache_params["save_rema"] and self.cache_params["use_cache"]:
                print(f"[INFO] Save cached modern elevation data")
                file_name = f"rema_{self.absolute_bounds[0]}_{self.absolute_bounds[1]}_" \
                    f"{self.absolute_bounds[2]}_{self.absolute_bounds[3]}_{zoom_lvl}m.tif"
                pth_cached_file= os.path.join(self.data_fld, "rema", file_name)
                os.makedirs(os.path.dirname(pth_cached_file), exist_ok=True)
                eti.export_tiff(self._dem_new, pth_cached_file,
                                overwrite=True,
                                transform=self._dem_new_transform,
                                use_lzw=True)

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
            print(f"[INFO] Load historical absolute DEM")
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
            print(f"[INFO] Load historical relative DEM")
            self._dem_old_rel, self._dem_old_rel_transform = li.load_image(self.pth_dem_rel,
                                                                   return_transform=True)
        return self._dem_old_rel, self._dem_old_rel_transform

    @property
    def dem_mask_new(self):

        if self._dem_mask_new is None:

            print("[INFO] Create mask for the modern DEM")

            # get the modern dem
            dem_new, _ = self.dem_new

            if self._dem_new is None:
                raise ValueError("The modern DEM is required to create the mask")

            self._dem_mask_new = np.ones(dem_new.shape, dtype=bool)

            if self.project_params["dem_mask"]["use_rock"]:
                self._dem_mask_new[self.rock_mask == 0] = 0

            if self.project_params["dem_mask"]["use_slope"]:
                print(f"[INFO] Use slope of {self.gcp_slope}° for the mask "
                      f"({self.active_slope})")

                if self.active_slope == "gcps":
                    input_slope = self.gcp_slope
                elif self.active_slope == "eval":
                    input_slope = self.eval_slope
                else:
                    raise ValueError("active_slope must be 'gcps' or 'eval'")

                # load slope and apply to mask
                self._dem_mask_new[self.slope_new > input_slope] = 0

        return self._dem_mask_new

    @property
    def dem_mask_old_abs(self):

        if self._dem_old_abs is None:
            raise ValueError("The absolute DEM is required to create the mask")

        if self._dem_mask_old_abs is None:

            print("[INFO] Create mask for the absolute DEM")

            self._dem_mask_old_abs = np.ones(self.dem_old_abs.shape, dtype=bool)

            # filter based on confidence
            if self.project_params["dem_mask"]["use_confidence"]:
                self._dem_mask_old_abs[self.conf_arr_abs <
                                       self.function_params["dem_mask"]["min_confidence"]] = 0

        return self._dem_mask_old_abs

    @property
    def dem_mask_old_rel(self):

        if self._dem_old_rel is None:
            raise ValueError("The relative DEM is required to create the mask")

        if self._dem_mask_old_rel is None:

            print("[INFO] Create mask for the relative DEM")

            # get the old dem_rel
            dem_old_rel, _ = self.dem_old_rel

            # create initial mask
            self._dem_mask_old_rel = np.ones(dem_old_rel.shape, dtype=bool)

            # filter based on confidence
            if self.project_params["dem_mask"]["use_confidence"]:
                self._dem_mask_old_rel[self.conf_arr_rel <
                         self.project_params["dem_mask"]["min_confidence"]] = 0

            if self.project_params["dem_mask"]["confidence_buffer_px"] > 0:
                buffer_px = self.project_params["dem_mask"]["confidence_buffer_px"]
                self._dem_mask_old_rel = binary_dilation(self._dem_mask_old_rel,
                                                         structure=np.ones((2 * buffer_px + 1, 2 * buffer_px + 1)))

            # release some memory
            del self._conf_arr_rel
            gc.collect()

        return self._dem_mask_old_rel

    @property
    def ortho_new(self):
        """
        Lazy-loads the modern ortho and its associated affine transform from
        Sentinel-2 if not already loaded. It will be downloaded automatically
        if not available locally.

        Returns:
            _ortho_new np.ndarray: The modern ortho.
            _ortho_new_transform: The affine transformation matrix
                associated with the ortho.
        Raises:
            ValueError: If `absolute_bounds` is not set.

        """
        if self.absolute_bounds is None:
            raise ValueError("Absolute bounds are required to load the modern ortho")

        if self._ortho_new is None:
            print("[INFO] Load modern ortho from Sentinel-2 with", self.absolute_bounds)
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

        if self.absolute_bounds is None:
            raise ValueError("Absolute bounds are required to load the rock mask")

        if self._rock_mask is None:

            print("[INFO] Load rock mask from Quantarctica with",
                  self.absolute_bounds)

            # load rock mask and apply to mask
            self._rock_mask = lrm.load_rock_mask(self.absolute_bounds,
                                           self.project_params["rema"]["zoom_level"],
                                           self.project_params["dem_mask"]["rock_buffer"])

        return self._rock_mask

    @property
    def slope_new(self) -> np.ndarray:
        """
        Lazy-loads and create the slope map of the new (=modern) DEM.

        Returns:
            _slope_new (np.ndarray): The slope of the new/modern DEM in degrees.
        """

        if self._slope_new is None:

            print("[INFO] Create slope map of the modern DEM")

            # get dem and transform
            d_new, t_new = self.dem_new
            self._slope_new = cs.create_slope(d_new, t_new,
                                              self.project_params["epsg_code"],
                                              self.project_params["no_data_val"])
        return self._slope_new

    def set_input_images(self, image_names: list[str],
                         src_folder: str | None = None,
                         extension: str = "tif") -> None:
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

        print("[INFO] Set {} input images.".format(len(image_names)))

        if len(image_names) < 3:
            raise ValueError("At least 3 images are required for SFM.")

        if src_folder is None:
            src_folder = DEFAULT_IMAGE_FLD

        # remove extension from image names
        image_names = [name.split(".")[0] for name in image_names]

        # save the images
        self.image_names = image_names

        # create subfolder for images
        image_folder = os.path.join(self.data_fld, "images")
        os.makedirs(image_folder, exist_ok=True)

        # copy images to the data folder
        for image_name in self.image_names:

            # add extension to image name
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
                          absolute_bounds: tuple[int, int, int, int] | None = None,
                          mask_src_folder: str | None = None,
                          ortho_new: np.ndarray | None =None,
                          ortho_new_transform: Affine | None = None,
                          dem_new: np.ndarray | None =None,
                          dem_new_transform: Affine | None = None,
                          camera_accuracies: dict[str, tuple[int, int, int]] | None = None,
                          camera_footprints: dict[str, BaseGeometry] | None = None,
                          camera_positions: dict[str, tuple[int, int, int]] | None = None,
                          camera_rotations: dict[str, tuple[int, int, int]] | None = None,
                          focal_lengths: dict[str, float] | None = None):

        # before optional data can be set, check if images are set
        if len(self.image_names) < 3:
            raise ValueError("Images must be set before setting input data.")

        # some attributes are required to be set together
        if ortho_new is not None and ortho_new_transform is None:
            raise ValueError("Ortho new transform must be set if ortho new is set.")
        if dem_new is not None and dem_new_transform is None:
            raise ValueError("DEM new transform must be set if DEM new is set.")

        # set some attributes directly
        if absolute_bounds is not None:
            self.absolute_bounds = absolute_bounds
        if ortho_new is not None:
            self._ortho_new = ortho_new
        if ortho_new_transform is not None:
            self._ortho_new_transform = ortho_new_transform
        if dem_new is not None:
            self._dem_new = dem_new
        if dem_new_transform is not None:
            self._dem_new_transform = dem_new_transform

        if mask_src_folder is not None:

            # get all tif files in the mask folder and only keep the ones that
            # are in the image folder
            all_masks = [str(p.resolve()) for p in Path(mask_src_folder).glob('*.tif')]

            # only keep masks that are in image folder
            masks = [mask for mask in all_masks if mask.split("/")[-1].split(".")[0] in self.image_names]
            masks = [Path(m).stem for m in masks]

            if len(masks) != len(self.image_names):
                raise ValueError("Number of masks must match number of images.")

            # define path to copy the masks
            mask_folder = os.path.join(self.data_fld, "masks_custom")

            # create folder
            os.makedirs(mask_folder, exist_ok=True)

            # copy masks to the mask folder
            for mask_name in masks:
                source_path = str(os.path.join(mask_src_folder, mask_name + ".tif"))
                dest_path = str(os.path.join(mask_folder, mask_name + ".tif"))
                shutil.copy(source_path, dest_path)

            self._update_data_availability_file("masks")

        if camera_accuracies is not None:
            if len(camera_accuracies) != len(self.image_names):
                raise ValueError("Number of camera accuracies must match "
                                 "number of images.")
            self.camera_accuracies = camera_accuracies
            self._update_data_availability_file("camera_accuracies")

        if camera_footprints is not None:
            if len(camera_footprints) != len(self.image_names):
                raise ValueError("Number of camera footprints must match "
                                 "number of images.")
            self.camera_footprints = camera_footprints
            self._update_data_availability_file("camera_footprints")

        if camera_positions is not None:

            if len(camera_positions) != len(self.image_names):
                raise ValueError("Number of camera positions must match "
                                 "number of images.")
            self.camera_positions = camera_positions
            self._update_data_availability_file("camera_positions")

        if camera_rotations is not None:
            if len(camera_rotations) != len(self.image_names):
                raise ValueError("Number of camera rotations must match "
                                 "number of images.")
            self.camera_rotations = camera_rotations
            self._update_data_availability_file("camera_rotations")

        if focal_lengths is not None:
            if len(focal_lengths) != len(self.image_names):
                raise ValueError("Number of focal lengths must match "
                                 "number of images.")
            self.focal_lengths = focal_lengths
            self._update_data_availability_file("focal_lengths")

        # use camera_footprints to possibly update the absolute bounds
        if self.data_availability['camera_footprints']:
            footprints = self.camera_footprints
            if self.project_params["bounding_box"]["only_vertical"]:
                # Filter polygons with "V" in the key
                footprints = {k: v for k, v in footprints.items() if "V" in k}
                if not footprints:
                    raise ValueError("No vertical camera footprints "
                                     "(with 'V' in the key) found.")

            # Combine all polygons into one & get the bounding box
            combined = unary_union(list(footprints.values()))
            absolute_footprint_bounds = combined.bounds
            if self.project_params["bounding_box"]["source"] == "footprints":
                self.absolute_bounds = absolute_footprint_bounds
            if self.project_params["bounding_box"]["source"]  == "combined":
                absolute_bounds = self.absolute_bounds
                if absolute_bounds is None:
                    self.absolute_bounds = absolute_footprint_bounds
                else:

                    # get the difference between the two bounds
                    diff_min_x = abs(absolute_bounds[0] - absolute_footprint_bounds[0])
                    diff_min_y = abs(absolute_bounds[1] - absolute_footprint_bounds[1])
                    diff_max_x = abs(absolute_bounds[2] - absolute_footprint_bounds[2])
                    diff_max_y = abs(absolute_bounds[3] - absolute_footprint_bounds[3])

                    # raise warning if any of the differences are larger than the warning value
                    warning_val = self.project_params["bounding_box"]["max_difference"]
                    if diff_min_x > warning_val or diff_min_y > warning_val or \
                        diff_max_x > warning_val or diff_max_y > warning_val:
                        print("[WARNING] The absolute bounds differ from the camera footprints by more than "
                              "{} m.".format(warning_val))
                        print("       ", self.absolute_bounds)
                        print("       ", absolute_footprint_bounds)

                    self.absolute_bounds = (min(absolute_bounds[0], absolute_footprint_bounds[0]),
                                            min(absolute_bounds[1], absolute_footprint_bounds[1]),
                                            max(absolute_bounds[2], absolute_footprint_bounds[2]),
                                            max(absolute_bounds[3], absolute_footprint_bounds[3]))
            self.data_availability["absolute_bounds"] = True

        # adapt the absolute bounds for rema
        if self.data_availability["absolute_bounds"]:
            rema_lvl = self.project_params["rema"]["zoom_level"]
            self.absolute_bounds = ab.adapt_bounds(self.absolute_bounds, rema_lvl)

    def run_project(self):
        """run project with all steps"""

        self.prepare_images()
        self.apply_rel_sfm()
        self.georef_project()
        self.apply_abs_sfm()
        self.correct_data()
        self.evaluate_project()

    def prepare_images(self) -> None:
        """
        Prepares images for Structure-from-Motion (SfM) processing.

        This includes:
        - Adding photos to the Metashape chunk.
        - Optionally setting focal lengths and fixed camera parameters.
        - Detecting fiducials and creating masks.
        - Saving original and adapted masks to disk.
        - Optionally enhancing images using provided masks.
        - Matching images via custom or built-in matching of Metashape.
        - Creating a bundler file and importing cameras if custom matching is enabled.

        Raises:
            FileNotFoundError: If any required input files (images or masks)
                are not located in the data folder.
            ValueError: If parameters are inconsistent or insufficient
                (e.g., missing or fewer than 3 images).
        """

        # check overwrite (no need checking for previous steps, as first step)
        if self.status_flags["prepare_images"]:
            if self.overwrite is False:
                print("[INFO] Images already prepared. Skipping preparation.")
                return

        print("[STEAP] Prepare images.")

        # check image availability
        if self.data_availability["images"] is False:
            raise ValueError("No images available. Please set images first.")

        # set path to the possible folders
        image_folder = os.path.join(self.data_fld, "images")
        enhanced_folder = os.path.join(self.data_fld, "images_enhanced")
        agi_mask_folder = os.path.join(self.data_fld, "masks_agi")
        adapted_mask_folder = os.path.join(self.data_fld, "masks_adapted")
        bundler_folder = os.path.join(self.data_fld, "bundler")

        # set lzw
        if self.project_params["compress_output"]:
            use_lzw= True
        else:
            use_lzw= False

        # get all images
        lst_image_paths = [str(p.resolve()) for p in Path(image_folder).glob('*.tif')]
        if len(lst_image_paths) == 0:
            raise FileNotFoundError("No images found in the image folder.")

        # TODO TEMP
        # add the images to the chunk
        #with redirect_output_to_file(self.pth_log):
        #    progress_callback = self._create_progress_bar("Add Photos")
        #    self.chunk.addPhotos(lst_image_paths, progress=progress_callback)

        #

        self.chunk.addPhotos(lst_image_paths, strip_extensions=True)

        # set the focal length of the images
        if self.data_availability["focal_lengths"]:
            for camera in self.chunk.cameras:

                # skip cameras that are not enabled
                if camera.enabled is False:
                    continue

                print(camera.label, self.focal_lengths[camera.label])

                # set the focal length
                camera.sensor.focal_length = self.focal_lengths[camera.label]

        # get fixed parameters as list
        fixed_params = [k for k, v in self.project_params["fixed_parameters"].items() if v]

        # set camera params to fixed
        for camera in self.chunk.cameras:
            # skip cameras that are not enabled
            if camera.enabled is False:
                continue
            camera.sensor.fixed_params = fixed_params

        # set images to film cameras (for finding fiducials)
        for camera in self.chunk.cameras:

            # skip cameras that are not enabled
            if camera.enabled is False:
                continue

            # set camera to film camera
            camera.sensor.film_camera = True
            camera.sensor.fixed_calibration = True

            # set pixel size
            pixel_size = self.project_params["pixel_size"]
            camera.sensor.pixel_size = (pixel_size, pixel_size)  # noqa

        # save all changes to camera
        self.doc.save()

        # find fiducials
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Detect Fiducials")
            arguments = self.agi_params["detectFiducials"]
            arguments["progress"] = progress_callback
            self.chunk.detectFiducials(**arguments)
            self.agi_pbar.set_postfix_str("- Finished -")
            self.agi_pbar.close()

        # check if we created masks for all images
        if arguments["generate_masks"]:
            self._update_data_availability_file("masks")

        # save agi masks to file
        if self.data_availability["masks"]:
            # create progressbar
            pbar = tqdm(total=len(self.image_names), desc="Save original masks",
                        position=0, leave=True)

            # save the original masks to file
            os.makedirs(agi_mask_folder, exist_ok=True)
            for camera in self.chunk.cameras:

                # update the progress bar description
                pbar.set_postfix_str(camera.label)

                # skip cameras that are not enabled
                if camera.enabled is False:
                    pbar.update(1)
                    continue

                # save the original mask to file
                mask = camera.mask.image()
                mask_bytes = mask.tostring()
                existing_mask = (np.frombuffer(mask_bytes, dtype=np.uint8).
                                 reshape((mask.height, mask.width)))
                pth_mask = os.path.join(agi_mask_folder, camera.label + ".tif")
                eti.export_tiff(existing_mask, pth_mask, use_lzw=use_lzw)

                # update progress bar
                pbar.update(1)

            # close progress bar
            pbar.set_postfix_str("- Finished -")
            pbar.close()

        # union masks to create adapted masks
        if self.data_availability["masks"]:

            # define path to the custom mask folder
            custom_mask_folder = os.path.join(self.data_fld, "masks_custom")

            # union masks
            arguments = self.function_params["union_masks_params"]
            um.union_masks(self.chunk, agi_mask_folder,
                           custom_mask_folder, adapted_mask_folder,
                           **arguments)
            self._update_data_availability_file("masks_adapted")

            tqdm.write("[INFO] Mask folder changed to {}".format(adapted_mask_folder))
            sleep(0.1)
            mask_folder = adapted_mask_folder
        else:
            mask_folder = agi_mask_folder

        # set cameras back to normal
        if self.project_params["film_cameras"] is False:
            for camera in self.chunk.cameras:
                # skip cameras that are not enabled
                if camera.enabled is False:
                    continue

                # set camera back to normal camera
                camera.sensor.film_camera = False
                camera.sensor.fixed_calibration = False

        # save changes to cameras
        self.doc.save()

        # enhance the images
        if self.project_params["enhance_images"]:

            arguments = {}
            if self.cache_params["use_cache"]:
                enhanced_cache_folder = os.path.join(self.cache_params["cache_folder"],
                                                     "enhanced")
                arguments["cache_folder"] = enhanced_cache_folder
                arguments["use_cached_images"] = self.cache_params["use_enhanced"]
                arguments["save_cached_images"] = self.cache_params["save_enhanced"]

            ei.enhance_images(enhanced_folder, image_folder,
                              mask_folder, self.image_names, **arguments)

            # after importing the cameras, set images to enhanced folder
            for camera in self.chunk.cameras:
                # skip cameras that are not enabled
                if camera.enabled is False:
                    continue

                # get image_path
                img_pth = os.path.join(enhanced_folder, camera.label + ".tif")

                # update camera path
                photo = camera.photo.copy()
                photo.path = img_pth
                camera.photo = photo

            # update the path to image folder to use the enhanced images
            image_folder = enhanced_folder

            self.doc.save()

            # set flag for enhanced images
            self._update_data_availability_file("images_enhanced")

        # no need for matching if there is a cached bundler file
        do_matching = True  # variable to check if we need to match the images
        if self.cache_params["use_cache"] and self.cache_params["use_bundler"]:
            bundler_cache_file_pth = os.path.join(self.cache_params["cache_folder"],
                                                  "bundler", f"{self.project_name}_bundler.txt")
            bundler_cache_hash_pth = os.path.join(self.cache_params["cache_folder"],
                                                  "bundler", f"{self.project_name}_bundler_hash.txt")
            # check if bundler file exists
            if os.path.exists(bundler_cache_file_pth):

                # calculate hash of image names
                sorted_list = sorted(self.image_names)
                list_str = json.dumps(sorted_list, separators=(',', ':'))
                hash_str = hashlib.sha256(list_str.encode('utf-8')).hexdigest()

                # check if hash file exists
                if os.path.exists(bundler_cache_hash_pth):
                    with open(bundler_cache_hash_pth, "r") as f:
                        bundler_hash = f.read().strip()

                    # if the hash matches, copy the bundler file to the bundler folder
                    if hash_str == bundler_hash:
                        os.makedirs(bundler_folder, exist_ok=True)
                        path_bundler_file = os.path.join(bundler_folder, "bundler.txt")

                        # copy bundler file to the bundler folder
                        shutil.copy(bundler_cache_file_pth, path_bundler_file)
                        do_matching = False

        # match the images
        if do_matching:
            if self.project_params["custom_matching"]:

                # get the combinations for matching
                arguments = self.function_params["combination_params"]
                combinations = cc.create_combinations(self.image_names,
                                                      footprint_dict=self.camera_footprints,
                                                      **arguments)

                # find tie points with our own custom matching
                arguments = self.function_params["custom_matching_params"]
                if self.debug:
                    arguments["debug"] = True
                    arguments["debug_folder"] = os.path.join(self.debug_fld, "tie_points")
                if self.cache_params["use_cache"]:
                    tp_cache_folder = os.path.join(self.cache_params["cache_folder"], "tie_points")
                    arguments["cache_folder"] = tp_cache_folder
                    arguments["use_cached_tps"] = self.cache_params["use_tps"]
                    arguments["save_cached_tps"] = self.cache_params["save_tps"]

                tp_dict, conf_dict = ftp.find_tie_points_for_sfm(image_folder, combinations,
                                                                 mask_folder=mask_folder,
                                                                 rotation_dict=self.camera_rotations,
                                                                 **arguments)

                # create the bundler file from the tie points
                os.makedirs(bundler_folder, exist_ok=True)
                arguments = self.function_params["create_bundler_params"]
                path_bundler_file = cb.create_bundler(image_folder,
                                                      bundler_folder,
                                                      tp_dict, conf_dict,
                                                      **arguments)

                # save the bundler file to the cache
                if self.cache_params["use_cache"] and self.cache_params["save_bundler"]:
                    bundler_cache_folder = os.path.join(self.cache_params["cache_folder"], "bundler")
                    os.makedirs(bundler_cache_folder, exist_ok=True)

                    # copy the bundler file
                    bundler_cache_file_pth = os.path.join(bundler_cache_folder,
                                                          f"{self.project_name}_bundler.txt")
                    shutil.copy(path_bundler_file, bundler_cache_file_pth)

                    # calculate hash of image names
                    sorted_list = sorted(self.image_names)
                    list_str = json.dumps(sorted_list, separators=(',', ':'))
                    hash_str = hashlib.sha256(list_str.encode('utf-8')).hexdigest()

                    # write hash to file
                    bundler_cache_hash_pth = os.path.join(bundler_cache_folder,
                                                            f"{self.project_name}_bundler_hash.txt")
                    with open(bundler_cache_hash_pth, "w") as f:
                        f.write(hash_str)

            else:

                # get the combinations for matching as pairs
                arguments = self.project_params["combinations_params"]
                arguments["pairwise"] = True
                pairs = cc.create_combinations(self.image_names,
                                                      self.camera_footprints,
                                                      **arguments)


                # match the images with agisoft
                with redirect_output_to_file(self.pth_log, self.debug):
                    progress_callback = self._create_progress_bar("Match Photos")
                    arguments = self.agi_params["matchPhotos"]
                    arguments["pairs"] = pairs
                    arguments["progress"] = progress_callback
                    self.chunk.matchPhotos(**arguments)
                    self.agi_pbar.set_postfix_str("- Finished -")
                    self.agi_pbar.close()

        # if custom matching import the bundler file
        if self.project_params["custom_matching"]:
            # import the bundler file
            with redirect_output_to_file(self.pth_log, self.debug):
                progress_callback = self._create_progress_bar("Import Cameras")
                self.chunk.importCameras(path_bundler_file,  # noqa
                                         format=Metashape.CamerasFormatBundler,
                                         progress=progress_callback)
                self.agi_pbar.set_postfix_str("- Finished -")
                self.agi_pbar.close()

        self.doc.save()

        # set status flag to true
        self._update_status_flag_file("prepare_images")

    def apply_rel_sfm(self) -> None:

        # check previous steps and also overwrite
        self._check_previous_steps("apply_rel_sfm")
        if self.status_flags["apply_rel_sfm"]:
            if self.overwrite is False:
                tqdm.write("[INFO] Relative SFM already applied. Skipping.")
                sleep(0.5)
                return

        tqdm.write("[STEP] Apply relative SFM.")
        sleep(0.5)

        # set lzw
        if self.project_params["compress_output"]:
            use_lzw= True
        else:
            use_lzw= False

        # align cameras
        arguments = self.agi_params["alignCamerasRelative"]
        # we need to reset the alignment if custom matching was used
        if self.project_params["custom_matching"]:
            arguments['reset_alignment'] = True
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Align Cameras")
            arguments["progress"] = progress_callback
            self.chunk.alignCameras(**arguments)
            self.agi_pbar.set_postfix_str("- Finished -")
            self.agi_pbar.close()

        self.doc.save()

        # check if cameras are aligned
        num_cameras = len(self.chunk.cameras)
        num_aligned = 0
        for camera in self.chunk.cameras:
            if camera.enabled and camera.transform:
                num_aligned += 1

        if num_aligned < 3:
            raise SfMError("Not enough cameras are aligned. "
                               "Please check the input images.")
        else:
            print(f"[INFO] {num_aligned}/{num_cameras} cameras aligned.")

        # build depth maps
        arguments = self.agi_params["buildDepthMapsRelative"]
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Build relative Depth Maps")
            arguments["progress"] = progress_callback
            self.chunk.buildDepthMaps(**arguments)
            self.doc.save()

        # build mesh
        arguments = self.agi_params["buildModelRelative"]
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Build relative Model")
            arguments["progress"] = progress_callback
            self.chunk.buildModel(**arguments)
            self.doc.save()

        # build point cloud
        arguments = self.agi_params["buildPointCloudRelative"]
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Build relative Point Cloud")
            arguments["progress"] = progress_callback
            self.chunk.buildPointCloud(**arguments)
            self.doc.save()

        # export point cloud
        arguments = self.agi_params["exportPointCloudRelative"]
        arguments['path'] = self.pth_pc_rel
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Export relative Point Cloud")
            arguments["progress"] = progress_callback
            self.chunk.exportPointCloud(**arguments)

        # we need to build already a bounding box for building DEM and ortho
        # this is based on the center and size of the chunk
        center = self.chunk.region.center
        size = self.chunk.region.size

        # Calculate the minimum and maximum corners of the bounding box
        min_corner = Metashape.Vector([center.x - size.x / 2,  # noqa
                                       center.y - size.y / 2,
                                       center.z - size.z / 2])
        max_corner = Metashape.Vector([center.x + size.x / 2,  # noqa
                                       center.y + size.y / 2,
                                       center.z + size.z / 2])
        # create 2d versions of the corners
        min_corner_2d = Metashape.Vector([min_corner.x, min_corner.y])  # noqa
        max_corner_2d = Metashape.Vector([max_corner.x, max_corner.y])  # noqa

        # temporary bbox
        temp_rel_bbox = Metashape.BBox(min_corner_2d, max_corner_2d)  # noqa

        # build DEM
        arguments = self.agi_params["buildDemRelative"]
        arguments["region"] = temp_rel_bbox
        arguments["resolution"] = self.project_params["resolution_relative"]
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Build relative DEM")
            arguments["progress"] = progress_callback
            self.chunk.buildDem(**arguments)
            self.doc.save()

        # build ortho
        arguments = self.agi_params["buildOrthoMosaicRelative"]
        arguments["region"] = temp_rel_bbox
        arguments["resolution_x"] = self.project_params["resolution_relative"]
        arguments["resolution_y"] = self.project_params["resolution_relative"]
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Build relative Ortho")
            arguments["progress"] = progress_callback
            self.chunk.buildOrthomosaic(**arguments)
            self.doc.save()

        # get relative bounds based on DEM and ortho
        ortho_left, ortho_right = self.chunk.orthomosaic.left, self.chunk.orthomosaic.right
        ortho_top, ortho_bottom = self.chunk.orthomosaic.top, self.chunk.orthomosaic.bottom
        dem_left, dem_right = self.chunk.elevation.left, self.chunk.elevation.right
        dem_top, dem_bottom = self.chunk.elevation.top, self.chunk.elevation.bottom
        self.relative_bounds = (min(ortho_left, dem_left),
                                min(ortho_top, dem_top),
                                max(ortho_right, dem_right),
                                max(ortho_bottom, dem_bottom))

        # export DEM
        arguments = self.agi_params["exportDemRelative"]
        arguments["path"] = self.pth_dem_rel
        arguments["save_alpha"] = False
        arguments["resolution_x"] = self.project_params["resolution_relative"]
        arguments["resolution_y"] = self.project_params["resolution_relative"]
        arguments["nodata_value"] = self.project_params["no_data_val"]
        arguments["region"] = temp_rel_bbox
        arguments["clip_to_boundary"] = True

        if self.project_params["compress_output"]:
            arguments["image_compression"] = self.compression
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Export relative DEM")
            arguments["progress"] = progress_callback
            self.chunk.exportRaster(**arguments)

        # export ortho
        arguments = self.agi_params["exportOrthoMosaicRelative"]
        arguments["path"] = self.pth_ortho_rel
        arguments["save_alpha"] = False
        arguments["resolution"] = self.project_params["resolution_relative"]
        arguments["region"] = temp_rel_bbox
        arguments["clip_to_boundary"] = True

        if self.project_params["compress_output"]:
            arguments["image_compression"] = self.compression
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Export relative Ortho")
            arguments["progress"] = progress_callback
            self.chunk.exportRaster(**arguments)

        # crop empty borders from output
        self.relative_bounds = co.crop_output(self.pth_dem_rel, self.pth_ortho_rel,
                               self.project_params["no_data_val"],255,
                               self.project_params["compress_output"],
                               self.project_params["epsg_code"],
                               self.project_params["crop_method"])

        # reset relative output values because we changed them
        self._dem_old_rel = None
        self._ortho_old_rel = None

        # build confidence arr
        d_rel, t_rel = self.dem_old_rel
        arguments = self.function_params["create_conf_arr_params"]
        arguments["no_data_val"] = self.project_params["no_data_val"]
        conf_arr_rel = cca.create_confidence_arr(d_rel, self.point_cloud_rel,
                                                 t_rel, **arguments)

        # export confidence arr
        eti.export_tiff(conf_arr_rel, self.pth_conf_arr_rel, use_lzw=use_lzw)

        # set status flag to true
        self._update_status_flag_file("apply_rel_sfm")

    def georef_project(self):
        # check previous steps and also overwrite
        self._check_previous_steps("georef_project")
        if self.status_flags["georef_project"]:
            if self.overwrite is False:
                print("[INFO] Geo-referencing already applied. Skipping.")
                return

        print("[STEP] Georef project.")

        # stop if we are missing absolute bounds or camera footprints
        if self.data_availability["absolute_bounds"] is False:
            raise ValueError("Missing absolute bounds for geo-referencing.")

        # create data folder for geo-referencing
        georef_data_folder = os.path.join(self.data_fld, "georef")

        # get ortho-photos
        ortho_old_rel, _ = self.ortho_old_rel
        ortho_new, ortho_new_transform = self.ortho_new

        # check if we have already a rotation cached
        rotation_georef = None
        if self.cache_params["use_cache"] and self.cache_params["use_rotation"]:
            pth_rot_file = os.path.join(self.data_fld, "georef", "rotation.txt")
            if os.path.exists(pth_rot_file):
                try:
                    with open(pth_rot_file, "r") as f:
                        val = float(f.read())
                        if not np.isnan(val):
                            rotation_georef = val
                            print("[INFO] Using cached rotation: {}".format(rotation_georef))
                except (Exception,):
                    print("[WARNING] Could not read rotation from cache.")

        # georeference the relative ortho
        arguments = self.function_params["georef_ortho_params"]
        if self.debug:
            arguments["debug"] = True
            arguments["debug_folder"] = os.path.join(self.debug_fld, "georef_ortho")
        arguments["data_folder"] = georef_data_folder
        tpl = go.georef_ortho(relative_ortho=ortho_old_rel,
                              absolute_ortho=ortho_new,
                              absolute_transform=ortho_new_transform,
                              input_rotation=rotation_georef,
                              **arguments)

        # extract values from return object
        transform_georef = tpl[0]
        bounds_georef = tpl[1]
        rotation_georef = tpl[2]

        if self.cache_params["use_cache"] and self.cache_params["save_rotation"]:
            pth_fld = os.path.join(self.data_fld, "georef")
            os.makedirs(pth_fld, exist_ok=True)
            pth_rot_file = os.path.join(pth_fld, "rotation.txt")
            with open(pth_rot_file, "w") as f:
                f.write(f"{rotation_georef:.6f}")

        # adapt the bounds for rema
        bounds_georef = ab.adapt_bounds(bounds_georef,
                                        self.project_params["rema"]["zoom_level"])

        # set the new absolute bounds based on the georeferenced ortho
        self.absolute_bounds = bounds_georef

        # reset modern input data
        self._ortho_new = None
        self._dem_new = None

        # get modern data again
        ortho_new, ortho_new_transform = self.ortho_new
        dem_new, _ = self.dem_new

        # get historic dem
        dem_old_rel, _ = self.dem_old_rel

        # find gcps
        arguments = self.function_params["find_gcps_params"]
        if self.debug:
            arguments["debug"] = True
            arguments["debug_folder"] = os.path.join(self.debug_fld, "gcps")
        arguments["data_folder"] = georef_data_folder
        if self.project_params["dem_mask"]["use_slope"]:
            arguments["slope"] = self.slope_new
            arguments["start_slope"] = self.project_params["dem_mask"]["slope_min"]
            arguments["end_slope"] = self.project_params["dem_mask"]["slope_max"]
            arguments["slope_step"] = self.project_params["dem_mask"]["slope_step"]
        arguments["no_data_val"] = self.project_params["no_data_val"]
        arguments["min_gcps"] = self.project_params["gcps"]["min_gcps"]
        gcps = fg2.find_gcps(ortho_old_rel, ortho_new,
                            dem_old_rel, dem_new,
                            rotation_georef,
                            ortho_new_transform,
                            self.project_params["resolution_relative"],
                            mask_rel=self.dem_mask_old_rel,
                            mask_rock=self.rock_mask,
                            **arguments)

        print("[INFO] Found {} GCPs.".format(gcps.shape[0]))

        if self.project_params["gcps"]["accuracy_m"]["z"] == "auto":
            base_z_acc = 1
            slope_factor = 3
            slope_rad = np.deg2rad(gcps['slope'])
            dynamic_z = base_z_acc + slope_factor * np.tan(slope_rad)
            gcps['accuracy_z'] = dynamic_z

        # set expected accuracy of the markers in px and m
        self.chunk.marker_projection_accuracy = self.project_params["gcps"]["accuracy_px"]

        # add gcps as markers
        arguments = self.function_params["add_markers_params"]
        arguments["accuracy_dict"] = self.project_params["gcps"]["accuracy_m"]
        print(arguments)
        am.add_markers(self.chunk, gcps,
                       epsg_code=self.project_params["epsg_code"],
                       **arguments)

        # set complete project to absolute coordinates
        epsg_code = self.project_params["epsg_code"]
        self.chunk.crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa
        self.chunk.camera_crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa
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

        print("[STEP] Apply absolute SFM.")

        # set lzw
        if self.project_params["compress_output"]:
            use_lzw= True
        else:
            use_lzw= False

        # build depth maps
        arguments = self.agi_params["buildDepthMapsAbsolute"]
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Build absolute Depth Maps")
            arguments["progress"] = progress_callback
            self.chunk.buildDepthMaps(**arguments)
            self.doc.save()

        # build mesh
        arguments = self.agi_params["buildModelAbsolute"]
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Build absolute Model")
            arguments["progress"] = progress_callback
            self.chunk.buildModel(**arguments)
            self.doc.save()

        # export Model
        arguments = self.agi_params["exportModelAbsolute"]
        arguments["crs"] = Metashape.CoordinateSystem(f"EPSG::{self.project_params['epsg_code']}")  # noqa
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Export absolute Model")
            arguments["progress"] = progress_callback
            self.chunk.exportModel(**arguments)

        # build point cloud
        arguments = self.agi_params["buildPointCloudAbsolute"]
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Build absolute Point Cloud")
            arguments["progress"] = progress_callback
            self.chunk.buildPointCloud(**arguments)
            self.doc.save()

        # exint cloud
        arguments = self.agi_params["exportPointCloudAbsolute"]
        arguments['path'] = self.pth_pc_abs
        arguments['crs'] = Metashape.CoordinateSystem(f"EPSG::{self.project_params['epsg_code']}")  # noqa
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Export absolute Point Cloud")
            arguments["progress"] = progress_callback
            self.chunk.exportPointCloud(**arguments)

        # build DEM
        arguments = self.agi_params["buildDemAbsolute"]
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Build absolute DEM")
            arguments["progress"] = progress_callback
            self.chunk.buildDem(**arguments)
            self.doc.save()

        # build ortho
        arguments = self.agi_params["buildOrthomosaicAbsolute"]
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Build absolute Ortho")
            arguments["progress"] = progress_callback
            self.chunk.buildOrthomosaic(**arguments)
            self.doc.save()

        # export DEM
        arguments = self.agi_params["exportDemAbsolute"]
        arguments['path'] = self.pth_dem_abs
        arguments["save_alpha"] = False
        arguments['resolution_x'] = self.project_params["resolution_absolute"]
        arguments['resolution_y'] = self.project_params["resolution_absolute"]
        arguments["nodata_value"] = self.project_params["no_data_val"]
        if self.project_params["compress_output"]:
            arguments["image_compression"] = self.compression
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Export absolute DEM")
            arguments["progress"] = progress_callback
            self.chunk.exportRaster(**arguments)

        # export ortho
        arguments = self.agi_params["exportOrthomosaicAbsolute"]
        arguments['path'] = self.pth_ortho_abs
        arguments["save_alpha"] = False
        arguments["resolution_x"] = self.project_params["resolution_absolute"]
        arguments["resolution_y"] = self.project_params["resolution_absolute"]
        if self.project_params["compress_output"]:
            arguments['compression'] = self.compression
        arguments["save_alpha"] = False
        with redirect_output_to_file(self.pth_log, self.debug):
            progress_callback = self._create_progress_bar("Export absolute Ortho")
            arguments["progress"] = progress_callback
            self.chunk.exportRaster(**arguments)

        # no data settings must be done extra for the ortho
        ortho_abs, ortho_abs_transform = li.load_image(self.pth_ortho_abs,
                                                       return_transform=True)
        ortho_abs[ortho_abs == 255] = self.project_params["no_data_val"]
        eti.export_tiff(ortho_abs, self.pth_ortho_abs,
                        transform=ortho_abs_transform,
                        overwrite=True, use_lzw=use_lzw)

        # build confidence arr
        d_abs, t_abs = self.dem_old_abs
        arguments = self.function_params["create_confidence_arr"]
        arguments["no_data_val"] = self.project_params["no_data_val"]
        conf_arr_abs = cca.create_confidence_arr(d_abs, self.point_cloud_abs, t_abs,
                                  **arguments)

        # export confidence arr
        eti.export_tiff(conf_arr_abs, self.pth_conf_arr_abs,
                        transform=t_abs, use_lzw=use_lzw)

        # set status flag to true
        self._update_status_flag_file("apply_abs_sfm")

    def correct_data(self):
        # check previous steps and also overwrite
        self._check_previous_steps("correct_data")
        if self.status_flags["correct_data"]:
            if self.overwrite is False:
                print("[INFO] Data already corrected. Skipping.")
                return

        print("[STEP] Correct data.")

        # set lzw
        if self.project_params["compress_output"]:
            use_lzw= True
        else:
            use_lzw= False

        # correct dem
        tpl = cd.correct_dem(self.dem_old_abs, self.dem_new, self.slope_new)

        self._dem_old_corrected = tpl[0]
        self._dem_old_corrected_transform = tpl[1]

        # export corrected DEM
        eti.export_tiff(self._dem_old_corrected, self.pth_dem_corrected,
                        self._dem_old_corrected_transform,
                        use_lzw=use_lzw)

        # set status flag to true
        self._update_status_flag_file("correct_data")

    def evaluate_project(self):
        # check previous steps and also overwrite
        self._check_previous_steps("evaluate_project")
        if self.status_flags["evaluate_project"]:
            if self.overwrite is False:
                print("[INFO] Project already evaluated. Skipping.")
                return

        print("[STEP] Evaluate project.")

        edq.estimate_dem_quality(self.dem_old_abs, self.dem_new)

        # set status flag to true
        self._update_status_flag_file("evaluate_project")

    def _create_progress_bar(self, desc: str = "Processing", total: int = 100):
        """
        Initializes a tqdm progress bar and returns a callback for Metashape progress tracking.
        """
        self.agi_pbar = tqdm(total=total, desc=desc, unit="%",
                         bar_format="{desc}: {percentage:3.0f}% |{bar}|"
                                    " [{elapsed}<{remaining}, {rate_fmt}]",
                             position=0, leave=True)

        def progress_callback(progress: float):
            self.agi_pbar.n = int(progress)
            self.agi_pbar.refresh()

        return progress_callback

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
        """
        Load data availability from disk if available, otherwise initialize
            dict with all steps set to False.

        Returns:
            dict[str, bool]: Dictionary of data availability flags.
        """
        if not os.path.exists(self.json_data_availability_path):
            print("[WARNING] Data availability file not found, "
                  "initializing default flags.")
            return {step: False for step in self.data_availability}

        with open(self.json_data_availability_path, "r") as f:
            return json.load(f)

    def _update_data_availability_file(self,
                                       data_type: str | None = None) -> None:

        if data_type is not None:
            if data_type not in self.data_availability:
                raise ValueError(f"Unknown data type: {data_type}")

            # Mark the current step as done
            self.data_availability[data_type] = True

        with open(self.json_data_availability_path, "w") as f:
            json.dump(self.data_availability, f, indent=2)  # type: ignore[arg-type]

    def _load_params(self, yaml_path: str) -> dict:
        """
        Load a YAML parameter file and resolve any Metashape constants.

        This function recursively parses the YAML structure and replaces any
        string starting with 'Metashape.' with the corresponding Metashape enum
        or constant.

        Args:
            yaml_path (str): Path to the YAML file.

        Returns:
            dict: Dictionary of parameters with resolved Metashape constants.
        """

        type_map = {
            "float": float,
            "int": int,
            "str": str,
        }

        def __resolve(obj):
            # Recursively resolve entries in a dictionary
            if isinstance(obj, dict):
                return {k: __resolve(v) for k, v in obj.items()}

            # Recursively resolve entries in a list
            elif isinstance(obj, list):
                return [__resolve(v) for v in obj]

            # If the value is a string we have different action
            elif isinstance(obj, str):
                if obj.startswith("Metashape."):
                    parts = obj.split(".")[1:]  # skip "Metashape"
                    attr = Metashape
                    for p in parts:
                        attr = getattr(attr, p)
                    return attr
                elif obj in type_map:
                    return type_map[obj]
                else:
                    return obj
            # Return the value unchanged if it's not a string we care about
            else:
                return obj

        # Load YAML file into a Python dictionary
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        return __resolve(raw)