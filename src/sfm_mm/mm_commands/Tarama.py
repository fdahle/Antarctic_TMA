"""Python module for Tarama in Micmac."""

# Library imports
import re
import json
from typing import Any

# Local imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class Tarama(BaseCommand):
    """
    Tarama is used to compute a rectified image
    """

    required_args = ["ImagePattern", "Orientation"]
    allowed_args = ["ImagePattern", "Orientation", "Zoom", "Repere", "Out", "ZMoy", "KNadir",
                    "IncMax", "UnUseAXC"]

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
        shell_string = f'Tarama {self.mm_args["ImagePattern"]} {self.mm_args["Orientation"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        shell_dict["Tarama"] = shell_string

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
        }

        # Regex patterns
        image_pattern = re.compile(r'TA : (OIS-Reech_.+?\.tif)')

        # Iterate over each line to extract and organize information
        for line in raw_output:
            image_info = image_pattern.search(line)
            if image_info:
                current_image = image_info.group(1)
                stats["images"].append(current_image)

        stats["total_images_processed"] = len(stats["images"])

        # Serialize the dictionary to a JSON string
        json_output = json.dumps(stats, indent=4)

        # Define path to save the json file
        json_path = f"{self.project_folder}/stats/{name}_stats.json"

        # Save json_output to a file
        with open(json_path, "w") as file:
            file.write(json_output)

        if self.debug:
            print(f"Tarama: Stats saved to {json_path}")

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

        # TODO

        pass
