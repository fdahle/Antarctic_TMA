import glob
import json
import os.path
import re

from src.sfm.mm_commands._base_command import BaseCommand


class ReSampFid(BaseCommand):

    required_args = ["ImagePattern", "ScanResolution"]
    allowed_args = ["ImagePattern", "ScanResolution"]

    def __init__(self, *args, **kwargs):
        # Initialize the base class with all arguments passed to ReSampFid
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the input parameters
        self.validate_mm_parameters()

    def build_shell_string(self):

        # build the shell command
        shell_string = f'ReSampFid "{self.mm_args["ImagePattern"]}" ' \
                       f'{self.mm_args["ScanResolution"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        return shell_string

    def extract_stats(self, raw_output):

        # Initialize statistics dictionary
        stats = {
            "total_images_processed": 0,
            "images": []
        }

        # Extract total number of images processed
        matches_line = re.search(r'".*\.tif": (\d+) matches.', raw_output)
        if matches_line:
            stats["total_images_processed"] = int(matches_line.group(1))

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

        # Serialize the dictionary to a JSON string
        json_output = json.dumps(stats, indent=4)

        # save json_output to a file
        with open(f"{self.project_folder}/stats/schnaps_stats.json", "w") as file:
            file.write(json_output)

    def validate_mm_parameters(self):

        # adapt the image pattern for glob
        image_pattern = self.mm_args['ImagePattern'].replace("/.*.", "/*.")

        # check if we get images with the image pattern in mm_args
        image_files = glob.glob(self.project_folder + "/" + image_pattern)
        if len(image_files) == 0:
            raise ValueError(f"No images found with pattern {self.mm_args['ImagePattern']}")

        if self.mm_args['ScanResolution'] <= 0:
            raise ValueError("ScanResolution must be greater than 0")

    def validate_required_files(self):

        # check for camera xml files
        if os.path.isfile(self.project_folder + "/MicMac-LocalChantierDescripteur.xml") is False:
            raise FileNotFoundError("MicMac-LocalChantierDescripteur.xml is missing")
        if os.path.isfile(self.project_folder + "/Ori-InterneScan/MeasuresCamera.xml") is False:
            raise FileNotFoundError("MeasuresCamera.xml is missing")
