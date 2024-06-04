# Package imports
import re
import json

# Custom imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class Tarama(BaseCommand):
    """
    Tarama is used to compute a rectified image
    """

    required_args = ["ImagePattern", "Orientation"]
    allowed_args = ["ImagePattern", "Orientation", "Zoom", "Repere", "Out", "ZMoy", "KNadir",
                    "IncMax", "UnUseAXC"]

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
        # nothing needs to be done before the execution
        pass

    def after_execution(self):
        # nothing needs to be done after the execution
        pass

    def build_shell_dict(self):

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

    def extract_stats(self, name, raw_output):

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

    def validate_mm_parameters(self):

        if "/" in self.mm_args["ImagePattern"]:
            raise ValueError("ImagePattern cannot contain '/'. Use a pattern like '*.tif' instead.")

    def validate_required_files(self):
        pass
