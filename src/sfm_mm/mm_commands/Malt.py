"""Python module for Malt in Micmac."""

# Library imports
import glob
import json
import os.path
import re
import shutil
from typing import Any

# Local imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class Malt(BaseCommand):
    """
    Compute the DEM
    """

    required_args = ["Mode", "ImagePattern", "Orientation"]
    allowed_args = ["Mode", "ImagePattern", "Orientation", "Master", "SzW", "CorMS", "UseGpu", "Regul",
                    "DirMEC", "DirOF", "UseTA", "ZoomF", "ZoomI", "ZPas", "Exe", "Repere", "NbVI", "HrOr",
                    "LrOr", "DirTA", "Purge", "DoMEC", "DoOrtho", "UnAnam", "2Ortho", "ZInc", "DefCor",
                    "CostTrans", "Etape0", "AffineLast", "ResolOrtho", "ImMNT", "ImOrtho", "ZMoy",
                    "Spherik", "WMI", "vMasqIm", "MasqImGlob", "IncMax", "BoxClip", "BoxTerrain", "ResolTerrain",
                    "RoundResol", "GCC", "EZA", "Equiv", "MOri", "MaxFlow", "SzRec", "Masq3D",
                    "NbProc", "PSIBN", "InternalNoIncid", "PtDebug"]

    lst_of_modes = ["Ortho", "UrbanMNE", "GeomImage"]

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

        # define the folders
        input_fld = os.path.join(self.project_folder, "MEC-Malt")
        output_fld = os.path.join(self.project_folder, "output")

        # Pattern to match files starting with 'Z_num' and ending with 'MALT.tif'
        dem_pattern = os.path.join(input_fld, "Z_Num*MALT.tif")
        xml_pattern = os.path.join(input_fld, "NuageImProf_STD*.xml")

        # Get a list of files matching the pattern
        dem_files = glob.glob(dem_pattern)
        xml_files = glob.glob(xml_pattern)

        # define the file names based on the project name
        project_name = self.project_folder.split("/")[-1]
        filename_dem = "DEM_" + project_name + ".tif"
        filename_xml = "malt_" + project_name + ".xml"

        # No files found matching the pattern
        if dem_files:
            # Get the most recent file by modification time
            most_recent_file = max(dem_files, key=os.path.getmtime)
            # Move the most recent file to the output folder
            shutil.copy(most_recent_file, os.path.join(output_fld, filename_dem))
        else:
            print("Malt: No DEM file found")

        if xml_files:
            # Get the most recent file by modification time
            most_recent_file = max(xml_files, key=os.path.getmtime)
            # Move the most recent file to the output folder
            shutil.copy(most_recent_file, os.path.join(self.project_folder, filename_xml))
        else:
            print("Malt: No XML file found")

        if self.debug:
            print(f"Malt: Output exported to {output_fld}")

    def build_shell_dict(self) -> dict[str, str]:
        """
        This function builds the shell command.
        Returns:
            dict[str, str]: Dictionary containing the command name and the command string.
        """

        shell_dict = {}

        # build the basic shell command
        shell_string = f'Malt {self.mm_args["Mode"]} "{self.mm_args["ImagePattern"]}" ' \
                       f'{self.mm_args["Orientation"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            # true and false must be lowercase
            if type(val) == bool:
                if val:
                    val = "true"
                else:
                    val = "false"

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        shell_dict["Malt"] = shell_string

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
            "steps": []
        }

        # Regex patterns
        image_pattern = re.compile(r'TA : (OIS-Reech_.+?\.tif)')
        step_pattern = re.compile(r'-------- BEGIN STEP,  , Num = (\d+), DeZoomTer = (\d+), DeZoomIm = (\d+)')  # noqa
        block_pattern = re.compile(r'DO ONE BLOC \[(\d+),(\d+)] \[(\d+),(\d+)] \[(\d+),(\d+)]')
        # correl_pattern = re.compile(r'Correl Calc, Begin Opt')
        result_pattern = re.compile(
            r'TCor (\d+\.\d+) CTimeC (\d+\.\d+) TOpt (\d+\.\d+) Pts , R2 (\d+\.\d+), RN (\d+\.\d+) Pts , '
            r'R-GEN (\d+\.\d+), Isol (\d+\.\d+) {2}PT {2}(\d+e\+\d+)')  # noqa

        # Iterate over each line to extract and organize information
        current_step = None
        for line in raw_output:
            image_info = image_pattern.search(line)
            if image_info:
                current_image = image_info.group(1)
                stats["images"].append(current_image)

            step_info = step_pattern.search(line)
            if step_info:
                if current_step:
                    stats["steps"].append(current_step)
                current_step = {
                    "step_num": int(step_info.group(1)),
                    "dezoom_ter": int(step_info.group(2)),
                    "dezoom_im": int(step_info.group(3)),
                    "blocks": []
                }

            block_info = block_pattern.search(line)
            if block_info and current_step:
                current_step["blocks"].append({
                    "start": [int(block_info.group(1)), int(block_info.group(2))],
                    "end": [int(block_info.group(3)), int(block_info.group(4))],
                    "offset": [int(block_info.group(5)), int(block_info.group(6))]
                })

            result_info = result_pattern.search(line)
            if result_info and current_step:
                current_step["blocks"][-1]["results"] = {
                    "tcor": float(result_info.group(1)),
                    "ctimec": float(result_info.group(2)),
                    "topt": float(result_info.group(3)),
                    "r2": float(result_info.group(4)),
                    "rn": float(result_info.group(5)),
                    "r_gen": float(result_info.group(6)),
                    "isol": float(result_info.group(7)),
                    "pt": float(result_info.group(8))
                }

        if current_step:
            stats["steps"].append(current_step)

        stats["total_images_processed"] = len(stats["images"])

        # Serialize the dictionary to a JSON string
        json_output = json.dumps(stats, indent=4)

        # Define path to save the json file
        json_path = f"{self.project_folder}/stats/{name}_stats.json"

        # Save json_output to a file
        with open(json_path, "w") as file:
            file.write(json_output)

        if self.debug:
            print(f"Malt: Stats saved to {json_path}")

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
