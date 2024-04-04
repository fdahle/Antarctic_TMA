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
DEFAULT_MASK_FLD = "/data_1/ATM/data_1/aerial/TMA/masked"
DEFAULT_RESAMPLED_MASK_FLD = "/data_1/ATM/data_1/aerial/TMA/masked_resampled"
DEFAULT_XML_FLD = "/data_1/ATM/data_1/sfm/xml/images/"
DEFAULT_TRANSFORM_FLD = "/data_1/ATM/data_1/georef/sat/"

# constant for the default camera folder
DEFAULT_CAM_FLD = "/data_1/ATM/data_1/sfm/xml/camera"


class SFMProject(object):
    # list of supported commands
    valid_commands = {
        "AperiCloud": {
            "desc": "Visualize relative orientation",
            "file": "mm_cmd.AperiCloud"
        },
        "Campari": {
            "desc": "Adjust camera calibration and orientation by bundle adjustment",
            "file": "mm_cmd.Campari"
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
        "Nuage2Ply": {
            "desc": "Convert dense cloud to PLY format",
            "file": "mm_cmd.Nuage2Ply"
        },
        "Schnaps": {
            "desc": "Reduce tie-points",
            "file": "mm_cmd.Schnaps"
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

    def set_images(self, image_ids: list[str], image_folder: Optional[str] = None,
                   copy_masks: bool = False, mask_folder: Optional[str] = None,
                   copy_resampled: bool = False, resampled_image_folder: Optional[str] = None,
                   copy_resampled_masks: bool = False, resampled_mask_folder=None,
                   copy_xml: bool = False, xml_folder: Optional[str] = None,
                   copy_transform: bool = False, transform_folder: Optional[str] = None,
                   skip_missing: bool = False, overwrite: bool = False,) -> None:
        """
        Sets the images for the project by copying them from source folders to the project folder,
        with options for copying associated masks, resampled images, and XML files.
        Args:
            image_ids (List[str]): List of image IDs to process.
            image_folder (Optional[str]): Folder where original images are stored.
                Defaults to DEFAULT_IMAGE_FLD.
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
            overwrite (bool): Whether to overwrite existing files at the destination. Defaults to False.
            skip_missing (bool): Whether to skip missing files without raising an exception. Defaults to False.
        Returns:
            None
        """

        if self.debug:
            print(f"Copy {image_ids}")

        # set default folders if not provided
        image_folder = image_folder or DEFAULT_IMAGE_FLD
        mask_folder = mask_folder or DEFAULT_MASK_FLD
        resampled_image_folder = resampled_image_folder or DEFAULT_RESAMPLED_IMAGE_FLD
        resampled_mask_folder = resampled_mask_folder or DEFAULT_RESAMPLED_MASK_FLD
        transform_folder = transform_folder or DEFAULT_TRANSFORM_FLD
        xml_folder = xml_folder or DEFAULT_XML_FLD

        # copy images to the project (Iterate over a copy of list to allow removal during iteration)
        for image_id in image_ids.copy():

            # Image paths
            old_img_path = os.path.join(image_folder, image_id + ".tif")
            new_img_path = os.path.join(self.project_path, "images_orig", image_id + ".tif")

            # check if image is existing in the original folder
            if os.path.isfile(old_img_path):

                # do not copy if images are already in the images-orig folder
                if not (os.path.isfile(new_img_path) and overwrite is False):
                    shutil.copyfile(old_img_path, new_img_path)

            # image is not existing
            else:
                if skip_missing:
                    image_ids.remove(image_id)  # Skip missing image
                    continue
                else:
                    raise FileNotFoundError(f"No image found at {old_img_path}")

            # Masks
            if copy_masks:

                # Mask paths
                old_mask_path = os.path.join(mask_folder, image_id + ".tif")
                new_mask_path = os.path.join(self.project_path, "masks_orig", image_id + ".tif")

                # check if mask is existing in the original folder
                if os.path.isfile(old_mask_path):
                    if not (os.path.isfile(new_mask_path) and overwrite is False):
                        shutil.copyfile(old_mask_path, new_mask_path)
                else:
                    if skip_missing:
                        print(f"No mask found at {old_mask_path}")
                    else:
                        raise FileNotFoundError(f"No mask found at {old_mask_path}")

            # Resampled images
            if copy_resampled:
                old_resampled_img_path = os.path.join(resampled_image_folder,
                                                      "OIS-Reech_" + image_id + ".tif")
                new_resampled_img_path = os.path.join(self.project_path, "images",
                                                      "OIS-Reech_" + image_id + ".tif")
                if os.path.isfile(old_resampled_img_path):
                    if not (os.path.isfile(new_resampled_img_path) and overwrite is False):
                        shutil.copyfile(old_resampled_img_path, new_resampled_img_path)
                else:
                    if skip_missing:
                        print(f"No resampled image found at {old_resampled_img_path}")
                    else:
                        raise FileNotFoundError(f"No resampled image found at {old_resampled_img_path}")

            # Resampled masks
            if copy_resampled_masks:
                old_resampled_mask_path = os.path.join(resampled_mask_folder,
                                                       "OIS-Reech_" + image_id + ".tif")
                new_resampled_mask_path = os.path.join(self.project_path, "masks",
                                                       "OIS-Reech_" + image_id + ".tif")
                if os.path.isfile(old_resampled_mask_path):
                    if not (os.path.isfile(new_resampled_mask_path) and overwrite is False):
                        shutil.copyfile(old_resampled_mask_path, new_resampled_mask_path)
                else:
                    if skip_missing:
                        print(f"No resampled mask found at {old_resampled_mask_path}")
                    else:
                        raise FileNotFoundError(f"No resampled mask found at {old_resampled_mask_path}")

            # XML files
            if copy_xml:
                old_xml_path = os.path.join(xml_folder, "MeasuresIm-" + image_id + ".tif.xml")
                new_xml_path = os.path.join(self.project_path, "Ori-InterneScan", "MeasuresIm-" + image_id + ".tif.xml")

                if os.path.isfile(old_xml_path):
                    if not (os.path.isfile(old_xml_path) and overwrite is False):
                        shutil.copyfile(old_xml_path, new_xml_path)
                else:
                    if skip_missing:
                        print(f"No XML file found at {old_xml_path}")
                    else:
                        raise FileNotFoundError(f"No XML file found at {old_xml_path}")

            if copy_transform:
                old_transform_path = os.path.join(transform_folder, image_id + "_transform.txt")
                new_transform_path = os.path.join(self.project_path, "transforms", image_id + ".txt")

                if os.path.isfile(old_transform_path):
                    if not (os.path.isfile(old_transform_path) and overwrite is False):
                        shutil.copyfile(old_transform_path, new_transform_path)
                else:
                    if skip_missing:
                        print(f"No transform found at {old_transform_path}")
                    else:
                        raise FileNotFoundError(f"No transform found at {old_transform_path}")

        self.image_ids = image_ids

    def start(self, mode: str, commands: None,
              micmac_args: dict = None,
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
            stat_folder:
            raw_folder:
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

        if mode == "manual" and commands is None:
            raise ValueError("Commands must be provided in manual mode")

        # check if there are enough images
        #if len(self.image_ids) < 3:
        #    raise ValueError("Need at least 3 images for SfM")

        # check if every image is existing and has an image xml
        #missing_images = []
        #missing_xml = []
        #for image_id in self.image_ids:
        #    if os.path.isfile(self.project_path + "/" + image_id + ".tif") is False and \
        #            os.path.isfile(self.project_path + "/images_orig/" + image_id + ".tif") is False:
         #       missing_images.append(image_id)
         #   if os.path.isfile(self.project_path + "/Ori-InterneScan/MeasuresIm-" + image_id + ".tif.xml") is False:
         #       missing_images.append(image_id)

        #if len(missing_images) > 0:
        #    raise Exception("Image is missing for following image ids: " + str(missing_images))
        #if len(missing_xml) > 0:
        #    raise Exception("Image xml is missing for following image ids: " + str(missing_xml))

        # check if there's a camera xml
        #if os.path.isfile(self.project_path + "/MicMac-LocalChantierDescripteur.xml") is False or \
        #        os.path.isfile(self.project_path + "/Ori-InterneScan/MeasuresCamera.xml") is False:
        #    raise Exception("Camera xml is missing")

        # execute the commands in order of the list
        for command in commands:

            # check if the command is supported
            if command not in self.valid_commands:
                raise Exception(f"Command '{command}' is not supported")

            # execute the command
            self._execute_command(command, micmac_args, print_all_output, save_stats, save_raw)

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

        # create folder for the stats
        if os.path.isdir(self.project_path + "/stats") is False:
            os.mkdir(self.project_path + "/stats")

        # create folder for Ori-Relative
        if os.path.isdir(self.project_path + "/Ori-Relative") is False:
            os.mkdir(self.project_path + "/Ori-Relative")

        # create folder for Ori-Tapas
        if os.path.isdir(self.project_path + "/Ori-Tapas") is False:
            os.mkdir(self.project_path + "/Ori-Tapas")

    def _execute_command(self, command_name, micmac_args, print_all_output=False,
                         save_stats=False, save_raw=False):

        try:
            # Dynamically import the module (file)
            module = importlib.import_module(f"src.sfm.mm_commands.{command_name}")

            # Get the class from the imported module
            mm_class = getattr(module, command_name)

            # check if there are custom args for the command
            if command_name in micmac_args:
                mm_args = micmac_args[command_name]
            else:
                mm_args = {}

            # Create an instance of the class
            mm_command = mm_class(self.project_path,
                                  mm_args=mm_args,
                                  print_all_output=print_all_output,
                                  save_stats=save_stats, save_raw=save_raw)

        except (ImportError, AttributeError) as e:
            # Handle the error if the module or class is not found
            if e is ImportError:
                raise ImportError(f"Could not find a command named '{command_name}'")
            else:
                raise e

        if command_name.endswith("Custom"):
            mm_command.execute_custom_cmd()
        else:
            mm_command.execute_shell_cmd(self.micmac_path)
