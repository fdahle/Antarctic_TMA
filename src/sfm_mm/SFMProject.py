import importlib
import os.path
import re
import shutil

from typing import Optional

import src.base.connect_to_database as ctd
import src.base.create_mask as cm
import src.export.export_tiff as ei
import src.load.load_image as li
import src.sfm_mm.snippets.create_camera_csv as ccc

"""Ideas:
- have a steps.txt file in the project folder, which contains the steps that were done
- create html file with shows stuff
"""

# constants for the default image folders
DEFAULT_IMAGE_FLD = "/data/ATM/data_1/aerial/TMA/downloaded"
DEFAULT_RESAMPLED_IMAGE_FLD = "/data/ATM/data_1/aerial/TMA/downloaded_resampled"
DEFAULT_MASK_FLD = "/data/ATM/data_1/aerial/TMA/masked"
DEFAULT_RESAMPLED_MASK_FLD = "/data/ATM/data_1/aerial/TMA/masked_resampled"
DEFAULT_XML_FLD = "/data/ATM/data_1/sfm/xml/images/"
DEFAULT_TRANSFORM_FLD = "/data/ATM/data_1/georef/sat/"

# constant for the default camera folder
DEFAULT_CAM_FLD = "/data/ATM/data_1/sfm/xml/camera"


class SFMProject(object):
    # list of supported commands
    # noinspection SpellCheckingInspection
    valid_commands = {
        "AperiCloud": {
            "desc": "Visualize relative orientation",
            "file": "mm_cmd.AperiCloud"
        },
        "Campari": {
            "desc": "Adjust camera calibration and orientation by bundle adjustment",
            "file": "mm_cmd.Campari"
        },
        "CenterBascule": {
            "desc": "Transform a purely relative orientation, as computed with Tapas, "
                    "in an absolute one",
            "file": "mm_cmd.CenterBascule"
        },
        "GCPBascule": {
            "desc": "Georeference model using Ground Control Points (GCPs)",
            "file": "mm_cmd.GCPBascule"
        },
        "GCPConvert": {
            "desc": "Convert GCP file formats",
            "file": "mm_cmd.GCPConvert"
        },
        "GCPCustom": {
            "desc": "Custom GCP finding",
            "file": "mm_cmd.GCPCustom"
        },
        "HomolFilterMasq": {
            "desc": "Filter tie-points",
            "file": "mm_cmd.HomolFilterMasq"
        },
        "Malt": {
            "desc": "Compute DEM",
            "file": "mm_cmd.Malt"
        },
        "OriConvert": {
            "desc": "Transform GPS data from text to MicMacXML format",
            "file": "mm_cmd.OriConvert"
        },
        "Nuage2Ply": {
            "desc": "Convert dense cloud to PLY format",
            "file": "mm_cmd.Nuage2Ply"
        },
        "Schnaps": {
            "desc": "Reduce tie-points",
            "file": "mm_cmd.Schnaps"
        },
        "ReduceCustom": {
            "desc": "Cut off borders of images",
            "file": "mm_cmd.ReduceCustom"
        },
        "ReSampFid": {
            "desc": "resample the images",
            "file": "mm_cmd.ReSampFid"
        },
        "Tapas": {
            "desc": "compute the relative orientation",
            "file": "mm_cmd.Tapas"
        },
        "Tapioca": {
            "desc": "Generate tie points between images",
            "file": "mm_cmd.Tapioca"
        },
        "TapiocaCustom": {
            "desc": "Custom tie points generation",
            "file": "mm_cmd.TapiocaCustom",
        },
        "Tarama": {
            "desc": "Optimize image matching for dense reconstruction",
            "file": "mm_cmd.Tarama"
        },
        "Tawny": {
            "desc": "Radiometric equalization of images",
            "file": "mm_cmd.Tawny"
        }
    }

    def __init__(self, project_name, project_folder, project_epsg=3031, micmac_path=None,
                 auto_enter=False, debug=False, resume=False, overwrite=False):

        # save settings
        self.debug = debug
        self.auto_enter = auto_enter

        # project settings
        self.project_name = project_name
        self.project_path = project_folder + "/" + project_name
        self.project_epsg = project_epsg

        # micmac settings
        self.micmac_path = micmac_path

        # check project settings
        if len(project_name) == 0:
            raise ValueError("Project name cannot be empty")
        if os.path.isdir(project_folder) is False:
            raise ValueError("Path to project folder is invalid")

        # check resume and overwrite
        if resume and overwrite:
            raise ValueError("Cannot resume and overwrite at the same time")

        # print settings if debug
        if debug:
            print("Resume: " + str(resume), "; Overwrite: " + str(overwrite))
            if resume and os.path.isdir(self.project_path):
                print(f"Resuming project at {self.project_path}")

        # set resume to false if the project folder does not exist
        if os.path.isdir(self.project_path) is False:
            resume = False

        # check overwrite settings
        if resume is False:
            if os.path.isdir(self.project_path):
                if overwrite is False:
                    raise FileExistsError(f"Project folder at '{self.project_path}' already exists")
                else:
                    # remove folder and all its content
                    shutil.rmtree(self.project_path)

            # otherwise create the project folder and create the project structure
            else:
                os.mkdir(self.project_path)
                if self.debug:
                    print(f"Project folder created at {self.project_path}")

        # already init some variables required for matching
        self.image_ids = []
        self.camera_name = ""

        # create the project structure
        self._create_project_structure()

    def set_camera(self, camera_name, camera_folder=None, overwrite=False):

        # define the camera folder
        if camera_folder is None:
            camera_folder = DEFAULT_CAM_FLD

        # define lcd paths
        old_lcd_path = camera_folder + "/" + camera_name + "-LocalChantierDescripteur.xml"
        new_lcd_path = self.project_path + "/MicMac-LocalChantierDescripteur.xml"

        # define mc paths
        old_mc_path = camera_folder + "/" + camera_name + "-MeasuresCamera.xml"
        new_mc_path = self.project_path + "/Ori-InterneScan/MeasuresCamera.xml"

        # check if the camera xml files are existing
        if os.path.isfile(old_lcd_path) is False:
            raise FileNotFoundError(f"LocalChantierDescripteur for {camera_name} is missing")

        if os.path.isfile(old_mc_path) is False:
            raise FileNotFoundError(f"MeasuresCamera for {camera_name} is missing")

        # copy the camera xml files
        if os.path.isfile(new_lcd_path) is False or overwrite:
            shutil.copyfile(old_lcd_path, new_lcd_path)
        if os.path.isfile(new_mc_path) is False or overwrite:
            shutil.copyfile(old_mc_path, new_mc_path)

        # save the camera name
        self.camera_name = camera_name

    def prepare_files(self, image_ids: list[str], image_folder: Optional[str] = None,
                      create_masks: bool = False, copy_masks: bool = False,
                      mask_folder: Optional[str] = None,
                      copy_resampled: bool = False, resampled_image_folder: Optional[str] = None,
                      copy_resampled_masks: bool = False, resampled_mask_folder=None,
                      copy_xml: bool = False, xml_folder: Optional[str] = None,
                      copy_transform: bool = False, transform_folder: Optional[str] = None,
                      create_camera_positions: bool = False,
                      create_image_thumbnails=False, create_mask_thumbnails=False,
                      ignore_missing: bool = False, overwrite: bool = False) -> None:
        """
        Sets the images for the project by copying them from source folders to the project folder,
        with options for copying associated masks, resampled images, and XML files.
        Args:
            image_ids (List[str]): List of image IDs to process.
            image_folder (Optional[str]): Folder where original images are stored.
                Defaults to DEFAULT_IMAGE_FLD.
            create_masks(bool): Whether to create new masks instead of using existing ones.
                Defaults to False.
            copy_masks (bool): Whether to copy mask files. Defaults to False.
            mask_folder (Optional[str]): Folder where mask files are stored.
                Defaults to DEFAULT_MASK_FLD.
            copy_resampled (bool): Whether to copy resampled images. Defaults to False.
            resampled_image_folder (Optional[str]): Folder where resampled images are stored.
                Defaults to DEFAULT_RESAMPLED_IMAGE_FLD.
            copy_resampled_masks (bool): Whether to copy resampled mask files. Defaults to False.
            resampled_mask_folder (Optional[str]): Folder where resampled mask files are stored.
                Defaults to DEFAULT_RESAMPLED_MASK_FLD.
            copy_xml (bool): Whether to copy XML files. Defaults to False.
            xml_folder (Optional[str]): Folder where XML files are stored.
                Defaults to DEFAULT_XML_FLD.
            copy_transform (bool): Whether to copy transform files. Defaults to False.
            transform_folder (Optional[str]): Folder where transform files are stored.
                Defaults to DEFAULT_TRANSFORM_FLD.
            create_camera_positions (bool): Whether to create camera positions. Defaults to False.
            create_image_thumbnails (bool): Whether to create thumbnails of the images. Defaults to False.
            create_mask_thumbnails (bool): Whether to create thumbnails of the masks. Defaults to False.
            overwrite (bool): Whether to overwrite existing files at the destination. Defaults to False.
            ignore_missing (bool): Whether to ignore missing files without raising an exception. Defaults to False.
        Returns:
            None
        """

        # copy the images and get the images that were copied
        self.image_ids = self.set_images(image_ids, image_folder,
                                         create_image_thumbnails, ignore_missing, overwrite)

        if copy_masks and create_masks:
            raise ValueError("Cannot copy masks and create masks at the same time")

        if create_masks:
            self.create_masks(self.image_ids, image_folder, create_mask_thumbnails,
                              ignore_missing, overwrite)
        if copy_masks:
            self.set_masks(self.image_ids, mask_folder, create_mask_thumbnails,
                           ignore_missing, overwrite)
        if copy_resampled:
            self.set_resampled_images(self.image_ids, resampled_image_folder,
                                      ignore_missing, overwrite)
        if copy_resampled_masks:
            self.set_resampled_masks(self.image_ids, resampled_mask_folder,
                                     ignore_missing, overwrite)
        if copy_xml:
            self.set_xml_files(self.image_ids, xml_folder,
                               ignore_missing, overwrite)
        if copy_transform:
            self.set_transforms(self.image_ids, transform_folder,
                                ignore_missing, overwrite)

        if create_camera_positions:
            self.set_camera_positions(self.image_ids, ignore_missing)

    def set_images(self, image_ids, image_folder=None,
                   create_image_thumbnails=False,
                   ignore_missing=True, overwrite=False):

        if self.debug:
            print("Copy images", end='')

        # set default image folder if not provided
        image_folder = image_folder or DEFAULT_IMAGE_FLD

        # copy image_ids list to avoid changing the original list
        image_ids_copy = image_ids.copy()

        # List to store missing images
        missing_images = []

        # copy images to the project folder
        for image_id in image_ids:

            # Image paths
            old_img_path = os.path.join(image_folder, image_id + ".tif")
            new_img_path = os.path.join(self.project_path, "images_orig", image_id + ".tif")

            # check if image is existing in the original folder
            if os.path.isfile(old_img_path):

                # do not copy if images are already in the images-orig folder
                if not (os.path.isfile(new_img_path) and overwrite is False):
                    shutil.copyfile(old_img_path, new_img_path)

                    # create thumbnails if requested
                    if create_image_thumbnails:
                        # create thumbnail path
                        thumbnail_path = os.path.join(self.project_path, "visuals", image_id + "_thumb.jpg")

                        # load the image
                        image = li.load_image(new_img_path)

                        # save as png
                        ei.export_tiff(image, thumbnail_path)

            # image is not existing
            else:

                # add to missing images list
                missing_images.append(image_id)

                # remove from image_ids list so that the following steps are not executed for that image id
                image_ids_copy.remove(image_id)

        if len(missing_images) > 0 and ignore_missing is False:
            raise FileNotFoundError("Missing images for the following image ids: ", missing_images)

        if self.debug:
            print("\rCopy images - finished")

        return image_ids_copy

    def create_masks(self, mask_ids=None, image_folder=None,
                     create_mask_thumbnails=False,
                     ignore_missing=True, overwrite=False):
        if self.debug:
            print("Create masks", end='')

        # set default ids if not provided
        mask_ids = mask_ids or self.image_ids

        # connect to db
        conn = ctd.establish_connection()

        # create string from ids
        image_id_string = "','".join(mask_ids)

        # get fiducial and text position
        sql_string = f"SELECT * FROM images_fid_points WHERE image_id IN ('{image_id_string}')"
        data_fid_points = ctd.execute_sql(sql_string, conn)
        sql_string = f"SELECT * FROM images_extracted WHERE image_id IN ('{image_id_string}')"
        data_extracted = ctd.execute_sql(sql_string, conn)

        # set default folders if not provided
        image_folder = image_folder or DEFAULT_IMAGE_FLD

        # List to store missing masks
        missing_masks = []

        # iterate over the copied image ids
        for image_id in mask_ids:

            try:
                # get the path to new mask
                new_mask_path = os.path.join(self.project_path, "masks_orig", image_id + ".tif")

                # do not create maks if masks are already in the masks_orig folder and overwrite is False
                if not (os.path.isfile(new_mask_path) and overwrite is False):

                    # get the image
                    image = li.load_image(image_folder + "/" + image_id + ".tif")

                    # Get the fid marks for the specific image_id
                    fid_marks_row = data_fid_points.loc[data_fid_points['image_id'] == image_id].squeeze()

                    # Create fid mark dict using dictionary comprehension
                    fid_dict = {str(i): (fid_marks_row[f'fid_mark_{i}_x'], fid_marks_row[f'fid_mark_{i}_y']) for i in
                                range(1, 5)}

                    # get the text boxes of the image
                    text_string = data_extracted.loc[data_extracted['image_id'] == image_id]['text_bbox'].iloc[0]

                    if len(text_string) > 0 and "[" not in text_string:
                        text_string = "[" + text_string + "]"

                    # create text-boxes list
                    text_boxes = [list(group) for group in eval(text_string.replace(";", ","))]

                    # create the mask
                    mask = cm.create_mask(image, fid_dict, text_boxes)

                    # save the mask
                    ei.export_tiff(mask, new_mask_path)

                    # create thumbnails if requested
                    if create_mask_thumbnails:
                        # create thumbnail path
                        thumbnail_path = os.path.join(self.project_path, "visuals",
                                                      image_id + "_mask_thumb.jpg")

                        # convert to 8-bit
                        mask = mask * 255

                        # save as png
                        ei.export_tiff(mask, thumbnail_path)
            except (Exception,):
                missing_masks.append(image_id)

        # check if images are missing
        if len(missing_masks) > 0:
            if ignore_missing:
                print("Missing masks for the following images: ", missing_masks)
            else:
                raise FileNotFoundError("Missing masks for the following images: ", missing_masks)

        if self.debug:
            print("\rCreate masks - finished")

    def set_masks(self, mask_ids=None, mask_folder=None,
                  create_mask_thumbnails=False,
                  ignore_missing=True, overwrite=False):

        if self.debug:
            print("Copy masks", end='')

        # set default ids if not provided
        mask_ids = mask_ids or self.image_ids

        # set default folders if not provided
        mask_folder = mask_folder or DEFAULT_MASK_FLD

        # List to store missing masks
        missing_masks = []

        # iterate over the copied image ids
        for image_id in mask_ids:

            # create old and new path to the masks
            old_mask_path = os.path.join(mask_folder, image_id + ".tif")
            new_mask_path = os.path.join(self.project_path, "masks_orig", image_id + ".tif")

            # check if mask is existing in the original folder
            if os.path.isfile(old_mask_path):

                # do not copy if masks are already in the masks_orig folder and overwrite is False
                if not (os.path.isfile(new_mask_path) and overwrite is False):
                    shutil.copyfile(old_mask_path, new_mask_path)

                    # create thumbnails if requested
                    if create_mask_thumbnails:
                        # create thumbnail path
                        thumbnail_path = os.path.join(self.project_path, "visuals",
                                                      image_id + "_mask_thumb.jpg")

                        # load the image
                        image = li.load_image(new_mask_path)

                        # convert to 8-bit
                        image = image * 255

                        # save as png
                        ei.export_tiff(image, thumbnail_path)

            else:
                missing_masks.append(image_id)

        # check if images are missing
        if len(missing_masks) > 0:
            if ignore_missing:
                print("Missing masks for the following images: ", missing_masks)
            else:
                raise FileNotFoundError("Missing masks for the following images: ", missing_masks)

        if self.debug:
            print("\rCopy masks - finished")

    def set_resampled_images(self, resampled_image_ids=None, resampled_image_folder=None,
                             ignore_missing=True, overwrite=False):

        if self.debug:
            print("Copy resampled images", end='')

        # set default ids if not provided
        resampled_image_ids = resampled_image_ids or self.image_ids

        # set default folders if not provided
        resampled_image_folder = resampled_image_folder or DEFAULT_RESAMPLED_IMAGE_FLD

        # List to store missing resampled images
        missing_resampled_images = []

        # iterate over the copied image ids
        for image_id in resampled_image_ids:

            # create old and new path to the resampled images
            old_resampled_img_path = os.path.join(resampled_image_folder,
                                                  "OIS-Reech_" + image_id + ".tif")
            new_resampled_img_path = os.path.join(self.project_path, "images",
                                                  "OIS-Reech_" + image_id + ".tif")

            # check if the resampled image is existing in the original folder
            if os.path.isfile(old_resampled_img_path):

                # do not copy if images are already in the images folder and overwrite is False
                if not (os.path.isfile(new_resampled_img_path) and overwrite is False):
                    shutil.copyfile(old_resampled_img_path, new_resampled_img_path)
            else:
                missing_resampled_images.append(image_id)

        if len(missing_resampled_images) > 0:
            if ignore_missing:
                print("Missing resampled images for the following images: ", missing_resampled_images)
            else:
                raise FileNotFoundError("Missing resampled images for the following images: ",
                                        missing_resampled_images)

        if self.debug:
            print("\rCopy resampled images - finished")

    def set_resampled_masks(self, resampled_mask_ids=None, resampled_mask_folder=None,
                            ignore_missing=True, overwrite=False):

        if self.debug:
            print("Copy resampled masks", end='')

        # set default ids if not provided
        resampled_mask_ids = resampled_mask_ids or self.image_ids

        # set default folders if not provided
        resampled_mask_folder = resampled_mask_folder or DEFAULT_RESAMPLED_MASK_FLD

        # List to store missing resampled masks
        missing_resampled_masks = []

        # iterate over the copied image ids
        for image_id in resampled_mask_ids:

            # create old and new path to the resampled masks
            old_resampled_mask_path = os.path.join(resampled_mask_folder,
                                                   "OIS-Reech_" + image_id + ".tif")
            new_resampled_mask_path = os.path.join(self.project_path, "masks",
                                                   "OIS-Reech_" + image_id + ".tif")

            # check if the resampled mask is existing in the original folder
            if os.path.isfile(old_resampled_mask_path):

                # do not copy if masks are already in the masks folder and overwrite is False
                if not (os.path.isfile(new_resampled_mask_path) and overwrite is False):
                    shutil.copyfile(old_resampled_mask_path, new_resampled_mask_path)
            else:
                missing_resampled_masks.append(image_id)

            if len(missing_resampled_masks) > 0:
                if ignore_missing:
                    print("Missing resampled masks for the following images: ", missing_resampled_masks)
                else:
                    raise FileNotFoundError("Missing resampled masks for the following images: ",
                                            missing_resampled_masks)

            if self.debug:
                print("\rCopy resampled masks - finished")

    def set_xml_files(self, xml_ids=None, xml_folder=None,
                      ignore_missing=True, overwrite=False):

        if self.debug:
            print("Copy XML files", end='')

        # set default ids if not provided
        xml_ids = xml_ids or self.image_ids

        # set default folders if not provided
        xml_folder = xml_folder or DEFAULT_XML_FLD

        # List to store missing XML files
        missing_xml_files = []

        # iterate over the copied image ids
        for image_id in xml_ids:

            # create old and new path to the XML files
            old_xml_path = os.path.join(xml_folder, "MeasuresIm-" + image_id + ".tif.xml")
            new_xml_path = os.path.join(self.project_path, "Ori-InterneScan", "MeasuresIm-" + image_id + ".tif.xml")

            # check if the XML file is existing in the original folder
            if os.path.isfile(old_xml_path):

                # do not copy if XML files are already in the Ori-InterneScan folder and overwrite is False
                if not (os.path.isfile(new_xml_path) and overwrite is False):
                    shutil.copyfile(old_xml_path, new_xml_path)
            else:
                missing_xml_files.append(image_id)

        if len(missing_xml_files) > 0:
            if ignore_missing:
                print("Missing xml files for the following images: ", missing_xml_files)
            else:
                raise FileNotFoundError("Missing xml files for the following images: ",
                                        missing_xml_files)

        if self.debug:
            print("\rCopy XML files - finished")

    def set_transforms(self, transform_ids=None, transform_folder=None,
                       ignore_missing=True, overwrite=False):

        if self.debug:
            print("Copy transform files", end='')

        # set default ids if not provided
        transform_ids = transform_ids or self.image_ids

        # set default folders if not provided
        transform_folder = transform_folder or DEFAULT_TRANSFORM_FLD

        # List to store missing transform files
        missing_transform_files = []

        # iterate over the copied image ids
        for image_id in transform_ids:

            # create old and new path to the transform files
            old_transform_path = os.path.join(transform_folder, image_id + "_transform.txt")
            new_transform_path = os.path.join(self.project_path, "transforms", image_id + ".txt")

            # check if the transform file is existing in the original folder
            if os.path.isfile(old_transform_path):

                # do not copy if transform files are already in the transforms folder and overwrite is False
                if not (os.path.isfile(new_transform_path) and overwrite is False):
                    shutil.copyfile(old_transform_path, new_transform_path)
            else:
                missing_transform_files.append(image_id)

        if len(missing_transform_files) > 0:
            if ignore_missing:
                print("Missing transform files for the following images: ", missing_transform_files)
            else:
                raise FileNotFoundError("Missing transform files for the following images: ",
                                        missing_transform_files)

        if self.debug:
            print("\rCopy transform files - finished")

    def set_camera_positions(self, image_ids, ignore_missing=True):

        if self.debug:
            print("Save camera positions as csv", end='')

        # call function to create the csv
        ccc.create_camera_csv(image_ids, self.project_path + "/cameras.csv", prefix="OIS-Reech_",
                              input_epsg=self.project_epsg, output_epsg=self.project_epsg,
                              skip_missing=ignore_missing)

        if self.debug:
            print("\rSave camera positions as csv - finished")

    def start(self, mode: str, commands: Optional[list[str]] = None,
              micmac_args: dict = None,
              skip_existing: bool = False,
              print_all_output: bool = False,
              save_stats: bool = False,
              save_raw: bool = False) -> None:
        """
        Starts the processing of the project by executing a list of MicMac commands.
        Args:
            mode (str):
            commands:
            micmac_args:
            print_all_output (bool): Whether to print all output to the console. Defaults to False.
            save_stats (bool): Whether to save stats of the MicMac functions to a json-file in the "stats" folder.
                Defaults to False.
            save_raw (bool): Whether to save the raw output of the MicMac functions to a text-file in the "stats"
                folder. Defaults to false
        Returns:

        """

        if mode == "complete":
            commands = ["ReSampFid", "Tapioca", "HomolFilterMasq", "Schnaps", "Campari", "AperiCloud", "Malt"]
        elif mode == "complete_custom":
            commands = ["ReSampFid", "TapiocaCustom", "HomolFilterMasq", "Schnaps", "Campari", "AperiCloud", "Malt"]
        elif mode == "manual":
            commands = commands
        else:
            raise ValueError(f"Mode '{mode}' is not supported")

        if mode == "manual" and (commands is None or len(commands) == 0):
            raise ValueError("Commands must be provided in manual mode")

        # execute the commands in order of the list
        for command_name in commands:

            # get the specif arguments for the command
            if command_name in micmac_args:
                cmd_args = micmac_args[command_name]
            else:
                cmd_args = {}

            # check if command is already in the stats
            if skip_existing and \
                    os.path.isfile(self.project_path + "/stats/" + command_name + "_stats.json"):
                print(f"Command '{command_name}' was already executed")
                continue

            # recognize suffixes with _X and remove them
            pattern = r'_.*$'
            command = re.sub(pattern, '', command_name)

            # check if the command is supported
            if command not in self.valid_commands:

                # different error message if the command had a suffix
                if command != command_name:
                    exc_string = f"Command '{command}' ({command_name}) is not supported"
                else:
                    exc_string = f"Command '{command}' is not supported"

                raise Exception(exc_string)

            # execute the command
            self._execute_command(command, command_name,
                                  cmd_args, print_all_output, save_stats, save_raw,
                                  auto_enter=self.auto_enter, debug=self.debug)

    def _create_project_structure(self):

        # create folder for homol
        if os.path.isdir(self.project_path + "/Homol") is False:
            os.mkdir(self.project_path + "/Homol")

        # create folder for the xml files
        if os.path.isdir(self.project_path + "/Ori-InterneScan") is False:
            os.mkdir(self.project_path + "/Ori-InterneScan")

        # create folder for images resampled
        if os.path.isdir(self.project_path + "/images_orig") is False:
            os.mkdir(self.project_path + "/images_orig")

        # create folder for images
        if os.path.isdir(self.project_path + "/images") is False:
            os.mkdir(self.project_path + "/images")

        # create folder for masks
        if os.path.isdir(self.project_path + "/masks_orig") is False:
            os.mkdir(self.project_path + "/masks_orig")

        # create folder for transforms
        if os.path.isdir(self.project_path + "/transforms") is False:
            os.mkdir(self.project_path + "/transforms")

        # create folder for resampled masks
        if os.path.isdir(self.project_path + "/masks") is False:
            os.mkdir(self.project_path + "/masks")

        # create folder for Ori-Relative
        if os.path.isdir(self.project_path + "/Ori-Relative") is False:
            os.mkdir(self.project_path + "/Ori-Relative")

        # create folder for Ori-Tapas
        if os.path.isdir(self.project_path + "/Ori-Tapas") is False:
            os.mkdir(self.project_path + "/Ori-Tapas")

        # create output folder
        if os.path.isdir(self.project_path + "/output") is False:
            os.mkdir(self.project_path + "/output")

        # create visuals folder
        if os.path.isdir(self.project_path + "/visuals") is False:
            os.mkdir(self.project_path + "/visuals")

        # create folder for the stats
        if os.path.isdir(self.project_path + "/stats") is False:
            os.mkdir(self.project_path + "/stats")

    def _execute_command(self, command, command_name, mm_args, print_all_output=False,
                         save_stats=False, save_raw=False, auto_enter=False, debug=False):

        print("Execute command:", command_name)

        try:
            # Dynamically import the module (file)
            module = importlib.import_module(f"src.sfm_mm.mm_commands.{command}")

            # Get the class from the imported module
            mm_class = getattr(module, command)

            # Create an instance of the class
            mm_command = mm_class(self.project_path,
                                  mm_args=mm_args,
                                  command_name=command_name,
                                  print_all_output=print_all_output,
                                  save_stats=save_stats, save_raw=save_raw,
                                  auto_enter=auto_enter, debug=debug)

        except (ImportError, AttributeError) as e:
            # Handle the error if the module or class is not found
            if e is ImportError:
                raise ImportError(f"Could not find a command named '{command}'")
            else:
                raise e

        if command.endswith("Custom"):
            mm_command.execute_custom_cmd()
        else:
            mm_command.execute_shell_cmd(self.micmac_path)
