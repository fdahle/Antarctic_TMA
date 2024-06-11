"""Python module for Campari in Micmac."""

# Package imports
import os
import glob
import json
import re
import shutil
from typing import Any

# Custom imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class Campari(BaseCommand):
    """
    Perform a bundle adjustment and a refinement of the camera calibration using the GCPs
    """
    required_args = ["ImagePattern", "InputOrientation", "OutputOrientation"]
    allowed_args = ["ImagePattern", "InputOrientation", "OutputOrientation", "SH", "GCP", "EmGPS", "GpsLa",
                    "SigmaTieP", "FactElimTieP", "CPI1", "CPI2", "FocFree", "PPFree", "AffineFree", "AllFree",
                    "DetGCP", "Visc", "ExpTxt", "ImMinMax", "DegAdd", "DegFree", "DRMax", "PoseFigee"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the mm_args
        self.validate_mm_args()

        # validate the input arguments
        self.validate_mm_parameters()

    def before_execution(self) -> None:
        """
        This function is called before the execution of the command.
        """

        # nothing needs to be done before the execution
        pass

    def after_execution(self) -> None:
        """
        This function is called after the execution of the command.
        """

        # nothing needs to be done after the execution
        pass

    def build_shell_dict(self) -> dict[str, str]:
        """
        This function builds the shell command.
        Returns:
            dict[str, str]: Dictionary containing the command name and the command string.
        """

        shell_dict = {}

        # build the basic shell command
        shell_string = f'Campari "{self.mm_args["ImagePattern"]}" ' \
                       f'{self.mm_args["InputOrientation"]} {self.mm_args["OutputOrientation"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        shell_dict["Campari"] = shell_string

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
            "images": [],
            "warnings": {
                "count": 0,
                "types": []
            }
        }

        # Regex patterns
        image_pattern = re.compile(r'NUM \d+ FOR (OIS-Reech_.+?\.tif)')
        guimbal_pattern = re.compile(r'NO GUIMBAL (OIS-Reech_.+?\.tif) (\d+(\.\d+)?)')
        residual_pattern = re.compile(
            r'RES:\[(OIS-Reech_.+?\.tif)]\[C] ER2 ([\d.\-]+) Nn ([\d.\-]+) Of '
            r'([\d.\-]+) Mul ([\d.\-]+) Mul-NN ([\d.\-]+) Time ([\d.\-]+)')
        warning_count_pattern = re.compile(r'\*\*\* There were (\d+) warnings of (\d+) different type')
        warning_type_pattern = re.compile(r'(\d+) occurence of warn type \[(.+?)]')

        # Iterate over each line to extract and organize information
        for line in raw_output:
            image_info = image_pattern.search(line)
            if image_info:
                current_image = image_info.group(1)
                stats["images"].append({
                    "name": current_image,
                    "guimbal": None,
                    "residuals": []
                })

            guimbal_info = guimbal_pattern.search(line)
            if guimbal_info:
                image_name = guimbal_info.group(1)
                guimbal = float(guimbal_info.group(2))
                for img in stats["images"]:
                    if img["name"] == image_name:
                        img["guimbal"] = guimbal
                        break

            residual_info = residual_pattern.search(line)
            if residual_info:
                image_name = residual_info.group(1)
                residuals = {
                    "ER2": float(residual_info.group(2)),
                    "Nn": float(residual_info.group(3)),
                    "Of": float(residual_info.group(4)),
                    "Mul": float(residual_info.group(5)),
                    "Mul_NN": float(residual_info.group(6)),
                    "Time": float(residual_info.group(7))
                }
                for img in stats["images"]:
                    if img["name"] == image_name:
                        img["residuals"].append(residuals)
                        break

            warning_count_info = warning_count_pattern.search(line)
            if warning_count_info:
                stats["warnings"]["count"] = int(warning_count_info.group(1))

            warning_type_info = warning_type_pattern.search(line)
            if warning_type_info:
                stats["warnings"]["types"].append({
                    "count": int(warning_type_info.group(1)),
                    "type": warning_type_info.group(2)
                })

        stats["total_images_processed"] = len(stats["images"])

        # Serialize the dictionary to a JSON string
        json_output = json.dumps(stats, indent=4)

        # Define path to save the json file
        json_path = f"{self.project_folder}/stats/{name}_stats.json"

        # Save json_output to a file
        with open(json_path, "w") as file:
            file.write(json_output)

        if self.debug:
            print(f"Campari: Stats saved to {json_path}")

    def validate_mm_parameters(self) -> None:
        """
        Validate the input parameters of the command.
        """

        if "/" in self.mm_args["ImagePattern"]:
            raise ValueError("ImagePattern cannot contain '/'. Use a pattern like '*.tif' instead.")

    def validate_required_files(self) -> None:
        """
        Validate the required files of the command.
        """

        # check all tif files in images-subfolder and copy them to the project folder if not already there
        homol_files = glob.glob(self.project_folder + "/images/*.tif")
        for file in homol_files:
            base_name = os.path.basename(file)

            if os.path.isfile(self.project_folder + "/" + base_name) is False:
                shutil.copy(file, self.project_folder + "/" + base_name)
