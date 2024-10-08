"""Python module for CenterBascule in Micmac."""

# Library imports
import json
import os.path
import re
from typing import Any

# Local imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class CenterBascule(BaseCommand):
    """
    The CenterBascule tool allows to transform a purely relative orientation,
    as computed with Tapas, in an absolute one.
    """

    required_args = ["ImagePattern", "OrientationIn", "LocalizationOfInformationCenters", "Out"]
    allowed_args = ["ImagePattern", "OrientationIn", "LocalizationOfInformationCenters", "Out",
                    "L1", "CalcV"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:

        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the mm_args
        self.validate_mm_args()

        # validate the input parameters
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
        shell_string = f'CenterBascule {self.mm_args["ImagePattern"]} ' \
                       f'{self.mm_args["OrientationIn"]} ' \
                       f'{self.mm_args["LocalizationOfInformationCenters"]} ' \
                       f'{self.mm_args["Out"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        shell_dict["CenterBascule"] = shell_string

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

        # Initialize variable
        current_image = None  # noqa

        # Iterate over each line to extract and organize information
        for line in raw_output:
            if line.startswith("NUM"):
                image_info = re.search(r'NUM \d+ FOR (OIS-Reech_.+?\.tif)', line)
                if image_info:
                    current_image = image_info.group(1)
                    stats["images"].append({
                        "name": current_image,
                        "guimbal": None,
                        "residual": None,
                    })

            if line.startswith("NO GUIMBAL"):
                guimbal_info = re.search(r'NO GUIMBAL (OIS-Reech_.+?\.tif) (\d+(\.\d+)?)', line)
                if guimbal_info:
                    image_name = guimbal_info.group(1)
                    guimbal = float(guimbal_info.group(2))
                    for img in stats["images"]:
                        if img["name"] == image_name:
                            print(guimbal)
                            img["guimbal"] = guimbal
                            break

            if line.startswith("Basc-Residual"):
                residual_info = re.search(r'Basc-Residual (.+?) \[([\d.\-]+),([\d.\-]+),([\d.\-]+)]', line)
                if residual_info:
                    image_name = residual_info.group(1)
                    residual = [float(residual_info.group(2)), float(residual_info.group(3)),
                                float(residual_info.group(4))]
                    for img in stats["images"]:
                        if img["name"] == image_name:
                            img["residual"] = residual
                            break

            if "*** There were" in line:
                warning_count_info = re.search(r'\*\*\* There were (\d+) warnings of (\d+) different type', line)
                if warning_count_info:
                    stats["warnings"]["count"] = int(warning_count_info.group(1))

            # noinspection SpellCheckingInspection
            if "occurence of warn type" in line:
                warning_type_info = re.search(r'(\d+) occurence of warn type \[(.+?)]', line)
                if warning_type_info:
                    stats["warnings"]["types"].append({
                        "count": int(warning_type_info.group(1)),
                        "type": warning_type_info.group(2),
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
            print(f"CenterBascule: Stats saved to {json_path}")

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

        # check orientation folder
        orientation_path = self.project_folder + "/Ori-" + self.mm_args["OrientationIn"]
        if os.path.isdir(orientation_path) is False:
            raise ValueError(f"Orientation folder '{orientation_path}' does not exist.")

        # check localization file
        localization_path = self.project_folder + "/Ori-" + self.mm_args["LocalizationOfInformationCenters"]
        if os.path.isdir(localization_path) is False:
            raise ValueError(f"Localization folder '{localization_path}' does not exist.")
