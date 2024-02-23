import importlib
import os.path
import shutil

from typing import Optional

"""Ideas:
- have a steps.txt file in the project folder, which contains the steps that were done
- create html file with shows stuff
"""

# constants for the default image folders
DEFAULT_IMAGE_FLD = "/data_1/ATM/data_1/aerial/TMA/downloaded"
DEFAULT_RESAMPLED_IMAGE_FLD = "/data_1/ATM/data_1/aerial/TMA/downloaded_resampled"
DEFAULT_MASK_FLD = ""
DEFAULT_XML_FLD = "/data_1/ATM/data_1/sfm/xml/images/"

# constant for the default camera folder
DEFAULT_CAM_FLD = "/data_1/ATM/data_1/sfm/xml/camera"

class SFMProject(object):

    # list of supported commands
    valid_commands = {
        "AperiCloud": {
            "desc": "visualize relative orientation",
            "file": "mm_cmd.AperiCloud"
        },
        "Campari": {
            "desc": "",
            "file": "mm_cmd.Campari"
        },
        "GCPBascule": {
            "desc": "",
            "file": "mm_cmd.GCPBascule"
        },
        "GCPConvert": {
            "desc": "",
            "file": "mm_cmd.GCPConvert"
        },
        "HomolFilterMasq": {
            "desc": "filter tie-points",
            "file": "mm_cmd.HomolFilterMasq"
        },
        "Malt": {
            "desc": "compute DEM",
            "file": "mm_cmd.Malt"
        },
        "Nuage2Ply": {
            "desc": "",
            "file": "mm_cmd.Nuage2Ply"
        },
        "Schnaps": {
            "desc": "reduce tie-points",
            "file": "mm_cmd.Schnaps"
        },
        "ReSampFid": {
            "desc": "resample the images",
            "file": "mm_cmd.ReSampFid"
        },
        "Tapas": {
            "desc": "compute relative orientation",
            "file": "mm_cmd.Tapas"
        },
        "Tapioca": {
            "desc": "",
            "file": "mm_cmd.Tapioca"
        },
        "Tapioca_custom": {
            "desc": "",
            "file": "mm_cmd.Tapioca_custom"
        },
        "Tarama": {
            "desc": "",
            "file": "mm_cmd.Tarama"
        },
        "Tawny": {
            "desc": "",
            "file": "mm_cmd.Tawny"
        }
    }

    def __init__(self, project_name, project_folder, micmac_path=None,
                 debug=False, resume=False, overwrite=False):

        # debug settings
        self.debug = debug

        # project settings
        self.project_name = project_name
        self.project_path = project_folder + "/" + project_name

        # micmac settings
        self.micmac_path = micmac_path

        # check project settings
        if len(project_name) == 0:
            raise Exception("Project name cannot be empty")
        if os.path.isdir(project_folder) is False:
            raise Exception("Path to project folder is invalid")

        # check resume and overwrite
        if resume and overwrite:
            raise Exception("Cannot resume and overwrite at the same time")

        # print settings if debug
        if debug:
            print("Resume: " + str(resume), "; Overwrite: " + str(overwrite))

        # check if the project folder exists
        if os.path.isdir(self.project_path):
            if resume:
                if self.debug:
                    print(f"Resuming project at {self.project_path}")

            elif overwrite is False:
                raise Exception(f"Project folder at '{self.project_path}' already exists")
            else:
                # remove folder and all its content
                shutil.rmtree(self.project_path)

        # otherwise create the project folder and create the project structure
        if resume is False:
            os.mkdir(self.project_path)
            if self.debug:
                print(f"Project folder created at {self.project_path}")

        # already init some variables required for matching
        self.image_ids = []
        self.camera_name = ""

        self._create_project_structure()


    def set_camera(self, camera_name, camera_folder=None):

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
        if os.path.isfile(old_lcd_path) is False or os.path.isfile(old_mc_path) is False:
            raise Exception(f"Camera xml for {camera_name} is missing")

        # copy the camera xml files
        shutil.copyfile(old_lcd_path, new_lcd_path)
        shutil.copyfile(old_mc_path, new_mc_path)

        # save the camera name
        self.camera_name = camera_name

    def set_images(self, image_ids: list[str], image_folder: Optional[str] = None,
                   copy_masks: bool = False, mask_folder: Optional[str] = None,
                   copy_resampled: bool = False, resampled_image_folder: Optional[str] = None,
                   copy_xml: bool = False, xml_folder: Optional[str] = None,
                   overwrite: bool = False, skip_missing: bool = False) -> None:
        """
        Sets the images for the project by copying them from source folders to the project folder,
        with options for copying associated masks, resampled images, and XML files.
        Args:
            image_ids (List[str]): List of image IDs to process.
            image_folder (Optional[str]): Folder where original images are stored.
                Defaults to DEFAULT_IMAGE_FLD.
            copy_masks (bool): Whether to copy mask files.
                Defaults to False.
            mask_folder (Optional[str]): Folder where mask files are stored.
                Defaults to DEFAULT_MASK_FLD.
            copy_resampled (bool): Whether to copy resampled images. Defaults to False.
            resampled_image_folder (Optional[str]): Folder where resampled images are stored.
                Defaults to DEFAULT_RESAMPLED_IMAGE_FLD.
            copy_xml (bool): Whether to copy XML files. Defaults to False.
            xml_folder (Optional[str]): Folder where XML files are stored.
                Defaults to DEFAULT_XML_FLD.
            overwrite (bool): Whether to overwrite existing files at the destination. Defaults to False.
            skip_missing (bool): Whether to skip missing files without raising an exception. Defaults to False.

        Returns:
            None
        """

        if self.debug:
            print(f"Set images with {image_ids}")

        if image_folder is None:
            image_folder = DEFAULT_IMAGE_FLD

        if mask_folder is None:
            mask_folder = DEFAULT_MASK_FLD

        if resampled_image_folder is None:
            resampled_image_folder = DEFAULT_RESAMPLED_IMAGE_FLD

        if xml_folder is None:
            xml_folder = DEFAULT_XML_FLD

        for image_id in image_ids.copy():  # Iterate over a copy of list to allow removal during iteration

            # Image paths
            old_img_path = os.path.join(image_folder, image_id + ".tif")
            new_img_path = os.path.join(self.project_path, image_id + ".tif")
            new_img_path2 = os.path.join(image_folder, "images_orig", image_id + ".tif")

            # check if image is existing in the original folder
            if os.path.isfile(old_img_path):

                # do not copy if images are already in the images-orig folder
                if os.path.isfile(new_img_path2) and (not os.path.isfile(new_img_path) or overwrite):
                    shutil.copyfile(old_img_path, new_img_path)
            else:
                if skip_missing:
                    image_ids.remove(image_id)  # Skip missing image
                    continue
                else:
                    raise Exception(f"No image found at {old_img_path}")

            # Masks
            if copy_masks:
                old_mask_path = os.path.join(mask_folder, image_id + ".tif")
                new_mask_path = os.path.join(self.project_path, "masks_orig", image_id + ".tif")
                if os.path.isfile(old_mask_path):
                    if not os.path.isfile(new_mask_path) or overwrite:
                        shutil.copyfile(old_mask_path, new_mask_path)

            # Resampled images
            if copy_resampled:
                old_resampled_img_path = os.path.join(resampled_image_folder, "OIS-Reech_" + image_id + ".tif")
                new_resampled_img_path = os.path.join(self.project_path, "OIS-Reech_" + image_id + ".tif")
                if os.path.isfile(old_resampled_img_path):
                    if not os.path.isfile(new_resampled_img_path) or overwrite:
                        shutil.copyfile(old_resampled_img_path, new_resampled_img_path)

                        # if we copied a resampled file, the original must be moved in the images_orig folder
                        shutil.move(new_img_path, os.path.join(self.project_path, "images_orig", image_id + ".tif"))

            # XML files
            if copy_xml:
                old_xml_path = os.path.join(xml_folder, "MeasuresIm-" + image_id + ".tif.xml")
                new_xml_path = os.path.join(self.project_path, "Ori-InterneScan", "MeasuresIm-" + image_id + ".tif.xml")

                print(old_xml_path, new_xml_path)

                if os.path.isfile(old_xml_path):
                    if not os.path.isfile(new_xml_path) or overwrite:
                        shutil.copyfile(old_xml_path, new_xml_path)

        self.image_ids = image_ids


    def start(self, mode: str, commands: None, save_stats: bool = False,
              use_custom_matching: bool = False,
              stats_folder: Optional[str] = None):
        """
        Starts the processing of the project by executing a list of MicMac commands.
        Args:
            commands:
            save_stats:
            stats_folder:

        Returns:

        """

        if mode == "complete":
            commands = ["ReSampFid", "Tapioca", "HomolFilterMasq", "Schnaps", "Campari", "AperiCloud", "Malt"]
        elif mode == "manual":
            commands = commands
        else:
            raise ValueError(f"Mode '{mode}' is not supported")

        if use_custom_matching:
            # replace Tapioca with Tapioca_custom
            commands = [command if command != "Tapioca" else "Tapioca_custom" for command in commands]

        # check if there are enough images
        if len(self.image_ids) < 3:
            raise Exception("Need at least 3 images for SfM")

        # check if every image is existing and has an image xml
        missing_images = []
        missing_xml = []
        for image_id in self.image_ids:
            if os.path.isfile(self.project_path + "/" + image_id + ".tif") is False and \
                    os.path.isfile(self.project_path + "/images_orig/" + image_id + ".tif") is False:
                missing_images.append(image_id)
            if os.path.isfile(self.project_path + "/Ori-InterneScan/MeasuresIm-" + image_id + ".tif.xml") is False:
                missing_images.append(image_id)

        if len(missing_images) > 0:
            raise Exception ("Image is missing for following image ids: " + str(missing_images))
        if len(missing_xml) > 0:
            raise Exception ("Image xml is missing for following image ids: " + str(missing_xml))

        # check if there's a camera xml
        if os.path.isfile(self.project_path + "/MicMac-LocalChantierDescripteur.xml") is False or \
                os.path.isfile(self.project_path + "/Ori-InterneScan/MeasuresCamera.xml") is False:
            raise Exception("Camera xml is missing")

        for command in commands:
            if command not in self.valid_commands:
                raise Exception(f"Command '{command}' is not supported")

            # execute the command
            self._execute_command(command, save_stats, stats_folder, mm_path=self.micmac_path)

    def _copy_files(self):
        pass

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

        # create folder for resampled masks
        if os.path.isdir(self.project_path + "/masks") is False:
            os.mkdir(self.project_path + "/masks")

        # create folder for the stats
        if os.path.isdir(self.project_path + "/stats") is False:
            os.mkdir(self.project_path + "/stats")

        # create folder for Ori-Relative
        if os.path.isdir(self.project_path + "/Ori-Relative") is False:
            os.mkdir(self.project_path + "/Ori-Relative")

        # create folder for Ori-Tapas
        if os.path.isdir(self.project_path + "/Ori-Tapas") is False:
            os.mkdir(self.project_path + "/Ori-Tapas")

    def _execute_command(self, command_name, save_stats, save_folder, mm_path=None):

        try:
            # Dynamically import the module (file)
            module = importlib.import_module(f"src.sfm.mm_commands.{command_name}")

            # Get the class from the imported module
            mm_class = getattr(module, command_name)

            # Create an instance of the class
            mm_command = mm_class(self.project_path)

        except (ImportError, AttributeError) as e:
            # Handle the error if the module or class is not found
            raise ImportError(f"Could not find a command named '{command_name}'. Error: {e}")

        args = {
            "ImagePattern": "*.*tif",
            "ScanResolution": 0.025
        }

        mm_command.execute_shell_cmd(args, mm_path)
