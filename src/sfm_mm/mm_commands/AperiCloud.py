"""Python module for AperiCloud in Micmac."""

# Library imports
import os
import glob
import shutil
import re
import json
from typing import Any

# Local imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class AperiCloud(BaseCommand):
    """
    AperiCloud is used for generating a visualization of and sparse 3D model and
    cameras position, as computed by Tapas
    """

    required_args = ["ImagePattern", "Orientation"]
    # noinspection SpellCheckingInspection
    allowed_args = ["ImagePattern", "Orientation", "ExpTxt", "Out", "Bin",
                    "RGB", "SeuilEc", "LimBsH", "WithPoints", "CalPerIm",
                    "Focs", "WithCam", "ColCadre", "ColRay", "SH"]

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

        # define the output folder
        output_fld = os.path.join(self.project_folder, "output")

        # define path to the orthophoto
        input_path = os.path.join(self.project_folder,
                                  "AperiCloud_" + self.mm_args['Orientation'] +
                                  "_" + self.mm_args['SH'][5:] + ".ply")
        output_path = os.path.join(output_fld, "visualization_" + self.mm_args["Orientation"] + ".ply")

        # Move the output file to the output folder
        shutil.copy(input_path, output_path)

        if self.debug:
            print(f"AperiCloud: Output exported to {output_fld}")

    def build_shell_dict(self) -> dict[str, str]:
        """
        This function builds the shell command.
        Returns:
            dict[str, str]: Dictionary containing the command name and the command string.
        """

        shell_dict = {}

        # build the basic shell command
        shell_string = f'AperiCloud "{self.mm_args["ImagePattern"]}" {self.mm_args["Orientation"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        shell_dict["AperiCloud"] = shell_string

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

        # Iterate over each line to extract statistics
        for line in raw_output:
            if line.startswith("/home"):
                image_info = re.search(r'/home/.*?/mm3d" PastDevlop  '  # noqa
                                       r'\./(OIS-Reech_.+?\.tif)', line)  # noqa
                if image_info:
                    current_image = image_info.group(1)
                    stats["images"].append({
                        "name": current_image,
                        "guimbal": None,
                    })

            if line.startswith("NO GUIMBAL"):
                guimbal_info = re.search(r'NO GUIMBAL (.+?) (\d\.\d+)', line)
                if guimbal_info:
                    image_name = guimbal_info.group(1)
                    guimbal = float(guimbal_info.group(2))
                    for img in stats["images"]:
                        if img["name"] == image_name:
                            img["guimbal"] = guimbal
                            break

            if "*** There were" in line:
                warning_count_info = re.search(r'\*\*\* There were (\d+) warnings of '
                                               r'(\d+) different type', line)
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

        # define path to save the json file
        json_path = f"{self.project_folder}/stats/{name}_stats.json"

        # save json_output to a file
        with open(json_path, "w") as file:
            file.write(json_output)

        if self.debug:
            print(f"AperiCloud: Stats saved to {json_path}")

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

        # check the orientation folder
        orientation_folder = self.project_folder + "/Ori-" + self.mm_args["Orientation"]
        if os.path.isdir(orientation_folder) is False:
            raise FileNotFoundError(f"No Orientation folder found at '{orientation_folder}'.")

        # check all tif files in images-subfolder and copy them to the project folder if not already there
        homol_files = glob.glob(self.project_folder + "/images/*.tif")
        for file in homol_files:
            base_name = os.path.basename(file)

            if os.path.isfile(self.project_folder + "/" + base_name) is False:
                shutil.copy(file, self.project_folder + "/" + base_name)

        if "SH" in self.mm_args:
            if os.path.isdir(self.project_folder + "/" + self.mm_args["SH"]) is False:
                raise FileNotFoundError(f"'No SH-folder found at "
                                        f"{self.project_folder + '/' + self.mm_args['SH']}' ")
