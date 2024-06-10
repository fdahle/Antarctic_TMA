"""Python module for Tawny in Micmac."""

# Package imports
import json
import os
import shutil
import re

# Custom imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class Tawny(BaseCommand):
    """
    Create an ortho-image
    """
    required_args = ["DataDirectory"]
    allowed_args = ["DataDirectory", "RadiomEgal", "DEq", "DEqXY", "AddCste", "DegRap", "DegRapXY",  # noqa
                    "RGP", "DynG", "ImPrio", "SzV", "CorThr", "NbPerlm", "L1F", "SatThresh", "Out"]  # noqa

    def __init__(self, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the mm_args
        self.validate_mm_args()

        # validate the input parameters
        self.validate_mm_parameters()

    def before_execution(self):
        """
        This function is called before the execution of the command.
        """

        # nothing needs to be done before the execution
        pass

    def after_execution(self):
        """
        This function is called after the execution of the command.
        """

        # define the folders
        input_fld = os.path.join(self.project_folder, "Ortho-MEC-Malt")
        output_fld = os.path.join(self.project_folder, "output")

        # define path to the orthophoto
        orthophoto = os.path.join(input_fld, self.mm_args['Out'])

        # define a file name based on the project name
        project_name = self.project_folder.split("/")[-1]
        filename = "orthophoto_" + project_name + ".tif"

        # Move the output file to the output folder
        shutil.copy(orthophoto, os.path.join(output_fld, filename))

        if self.debug:
            print(f"Tawny: Output exported to {output_fld}")

    def build_shell_dict(self):
        """
        This function builds the shell command.
        """

        shell_dict = {}

        # build the basic shell command
        shell_string = f'Tawny {self.mm_args["DataDirectory"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        shell_dict["Tawny"] = shell_string

        return shell_dict

    def extract_stats(self, name, raw_output):
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
            "kboxes": []
        }

        # Regex patterns
        image_pattern = re.compile(r'Image (Ort_.+?\.tif)')
        kbox_pattern = re.compile(r'KBOX = (\d+) On (\d+)')

        # Iterate over each line to extract and organize information
        for line in raw_output:
            image_info = image_pattern.search(line)
            if image_info:
                current_image = image_info.group(1)
                stats["images"].append(current_image)

            kbox_info = kbox_pattern.search(line)
            if kbox_info:
                kbox_id = int(kbox_info.group(1))
                kbox_on = int(kbox_info.group(2))
                stats["kboxes"].append({
                    "kbox_id": kbox_id,
                    "on": kbox_on
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
            print(f"Tawny: Stats saved to {json_path}")

    def validate_mm_parameters(self):
        """
        Validate the input parameters of the command.
        """

        # TODO

        pass

    def validate_required_files(self):
        """
        Validate the required files of the command.
        """

        # TODO

        pass
