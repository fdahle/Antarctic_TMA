"""Python module for ReSampFid Micmac."""

# Library imports
import glob
import json
import os.path
import re
import shutil
from typing import Any

# Local imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class ReSampFid(BaseCommand):

    """
    ReSampFid is a tool to resample images using fiducial marks
    """

    required_args = ["ImagePattern", "ScanResolution"]
    allowed_args = ["ImagePattern", "ScanResolution"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Initialize the base class with all arguments passed to ReSampFid
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # set default values for the mm_args
        if "ResampleMasks" not in self.mm_args:
            self.mm_args["ResampleMasks"] = False

        # remove custom arguments from the arguments
        self.resample_masks = self.mm_args.pop("ResampleMasks")

        # validate the mm_args
        self.validate_mm_args()

        # validate the input parameters
        self.validate_mm_parameters()

    def before_execution(self) -> None:
        """
        This function is called before the execution of the command.
        """

        if self.debug:
            print("ResampFid: copy images")

        # copy the images to the main project folder
        for filename in os.listdir(self.project_folder + "/images_orig"):
            if not os.path.isfile(self.project_folder + "/" + filename):
                shutil.copyfile(self.project_folder + "/images_orig/" + filename,
                                self.project_folder + "/" + filename)

        # copy also the masks if ResampleMasks is True
        if self.resample_masks:

            if self.debug:
                print("ResampFid: copy masks")

            for filename in os.listdir(self.project_folder + "/masks_orig"):
                if not os.path.isfile(self.project_folder + "/" + "mask_" + filename):
                    # copy the mask
                    shutil.copyfile(self.project_folder + "/masks_orig/" + filename,
                                    self.project_folder + "/" + "mask_" + filename)
                    # copy the xml file to have one for each mask
                    # noinspection SpellCheckingInspection
                    shutil.copyfile(self.project_folder + f"/Ori-InterneScan/MeasuresIm-{filename}.xml",
                                    self.project_folder + f"/Ori-InterneScan/MeasuresIm-mask_{filename}.xml")

    def after_execution(self) -> None:
        """
        This function is called after the execution of the command.
        """

        if self.debug:
            print("ResampFid: remove temporary images")

        # remove the copied images
        for filename in os.listdir(self.project_folder + "/images_orig"):
            if os.path.isfile(self.project_folder + "/" + filename):
                os.remove(self.project_folder + "/" + filename)

        # remove the copied masks if ResampleMasks is True
        if self.resample_masks:
            for filename in os.listdir(self.project_folder + "/masks_orig"):
                if os.path.isfile(self.project_folder + "/" + "mask_" + filename):
                    # remove the mask
                    os.remove(self.project_folder + "/" + "mask_" + filename)

                    # remove the xml file
                    os.remove(self.project_folder + f"/Ori-InterneScan/"
                                                    f"MeasuresIm-mask_{filename}.xml")
                    os.remove(self.project_folder + f"/Ori-InterneScan/"
                                                    f"OIS-Reech_mask_{filename[:-4]}_ChambreMm2Pix.xml")

                    # copy the resampled mask to the masks folder
                    shutil.move(self.project_folder + "/OIS-Reech_mask_" + filename,
                                self.project_folder + "/masks/OIS-Reech_" + filename)

    def build_shell_dict(self) -> dict[str, str]:
        """
        This function builds the shell command.
        Returns:
            dict[str, str]: Dictionary containing the command name and the command string.
        """

        shell_dict = {}

        # build the shell command
        shell_string = f'ReSampFid "{self.mm_args["ImagePattern"]}" ' \
                       f'{self.mm_args["ScanResolution"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        # save the shell command
        shell_dict["ReSampFid"] = shell_string

        # create additional shell command for masks
        if self.resample_masks:
            # build the shell command
            shell_string = f'ReSampFid "mask_.*tif" ' \
                           f'{self.mm_args["ScanResolution"]}'

            # add the optional arguments to the shell string
            for key, val in self.mm_args.items():

                # skip required arguments
                if key in self.required_args:
                    continue

                shell_string = shell_string + " " + str(key) + "=" + str(val)

            shell_dict["ReSampFid_masks"] = shell_string

        return shell_dict

    def extract_stats(self, name: str, raw_output: list[str]) -> None:
        """
        Extract statistics from the raw output of the command and save them to a JSON file.
        Args:
            name (str): Name of the command.
            raw_output (list): Raw output of the command as a list of strings (one per line).
        Returns:
            None
        """

        # Initialize statistics dictionary
        stats = {
            "total_images_processed": 0,
            "images": []
        }

        # Iterate over each line to extract and organize information
        for line in raw_output:
            if line.startswith("==="):
                image_info = re.search(r"=== RESAMPLE EPIP (.+?) Ker=(\d+) Step=(\d+) SzRed=\[(\d+),(\d+)]======",
                                       line)
                if image_info:
                    image_name = image_info.group(1)
                    ker = int(image_info.group(2))
                    step = int(image_info.group(3))
                    szred = [int(image_info.group(4)), int(image_info.group(5))]

            if line.startswith("FOR"):
                residu_time_info = re.search(r"FOR (.+?) RESIDU (.+?) Time (.+?) ", line)
                if residu_time_info:
                    residu = float(residu_time_info.group(2))
                    time = float(residu_time_info.group(3))
                    # Ensure the image name matches between sections
                    if residu_time_info.group(1) == image_name:  # noqa
                        stats["images"].append({
                            "name": image_name,
                            "ker": ker,  # noqa
                            "step": step,  # noqa
                            "szred": szred,  # noqa
                            "residu": residu,  # noqa
                            "time": time,
                        })

        stats["total_images_processed"] = len(stats["images"])

        # Serialize the dictionary to a JSON string
        json_output = json.dumps(stats, indent=4)

        # define path to save the json file
        json_path = f"{self.project_folder}/stats/{name}_stats.json"

        # save json_output to a file
        with open(json_path, "w") as file:
            file.write(json_output)

        if self.debug:
            print(f"ReSampFid: Stats saved to {json_path}")

    def validate_mm_parameters(self) -> None:
        """
        Validate the input parameters of the command.
        """

        if self.debug:
            print("ReSampFid: Validate mm parameters", end='')

        # adapt the image pattern for glob
        image_pattern_glob = self.mm_args['ImagePattern'].replace(".*.", "*.")

        # check if we get images with the image pattern in mm_args
        image_files = glob.glob(self.project_folder + "/images_orig/" + image_pattern_glob)

        if len(image_files) == 0:
            raise ValueError(f"No images found with pattern {self.project_folder + '/' + image_pattern_glob}")

        if self.mm_args['ScanResolution'] <= 0:
            raise ValueError("ScanResolution must be greater than 0")

        if self.debug:
            print("\rReSampFid: Validate mm parameters - finished")

    def validate_required_files(self) -> None:
        """
        Validate the required files of the command.
        """

        if self.debug:
            print("ReSampFid: Validate required files", end='')

        # check for camera xml files
        if os.path.isfile(self.project_folder + "/MicMac-LocalChantierDescripteur.xml") is False:
            raise FileNotFoundError("MicMac-LocalChantierDescripteur.xml is missing")
        if os.path.isfile(self.project_folder + "/Ori-InterneScan/MeasuresCamera.xml") is False:
            raise FileNotFoundError("MeasuresCamera.xml is missing")

        # check for image xml files
        image_files = glob.glob(self.project_folder + "/images_orig/*.tif")
        for image_file in image_files:
            image_name = os.path.basename(image_file)[:-4]
            if os.path.isfile(self.project_folder + f"/Ori-InterneScan/MeasuresIm-{image_name}.tif.xml") is False:
                raise FileNotFoundError(f"MeasuresIm-{image_name}.tif.xml is missing")

        if self.debug:
            print("\rReSampFid: Validate required files - finished")
