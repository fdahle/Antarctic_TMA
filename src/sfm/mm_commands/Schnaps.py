import os
import glob
import json
import re
import shutil

from src.sfm.mm_commands._base_command import BaseCommand


class Schnaps(BaseCommand):
    required_args = ["ImagePattern"]
    allowed_args = ["ImagePattern", "HomolIn", "NbWin", "ExeWrite", "HomolOut", "ExpTxt",
                    "VeryStrict", "ShowStats", "DoNotFilter", "PoubelleName",
                    "minPercentCoverage", "MoveBadImgs", "OutTrash", "MiniMulti",
                    "NetworkExport"]

    def __init__(self, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the input parameters
        self.validate_mm_parameters()

        # validate the required files (e.g. for copying to the main project folder)
        self.validate_required_files()

    def build_shell_string(self):

        # build the basic shell command
        shell_string = f'Schnaps "{self.mm_args["ImagePattern"]}"'

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
            "general": {
                "homol_points_total": None,
                "homol_points_bad": None,
                "homol_points_bad_percentage": None,
                "pictures_rejected": None,
            },
            "sizes": [],
            "pictures": []
        }

        # Iterate over each line to extract and organize information
        for line in raw_output:
            # extract general information
            if "Homol points" in line:
                parts = re.findall(r'\d+', line)
                if len(parts) >= 2:
                    stats["general"].update({
                        "homol_points_total": int(parts[0]),
                        "homol_points_bad": int(parts[1]),
                        "homol_points_bad_percentage": float(parts[2]) if len(parts) > 2 else None,
                    })
            elif "pictures rejected" in line:
                # Extract the number of rejected pictures
                rejected_pictures_match = re.search(r'(\d+)\s+pictures rejected', line)
                if rejected_pictures_match:
                    stats["general"]["pictures_rejected"] = int(rejected_pictures_match.group(1))  # noqa
            # extract sizes information
            elif line.strip().startswith("* ["):
                size_info = re.findall(r'\[\d+,\d+]', line)[0]
                windows_info = re.findall(r'\[\d+,\d+]', line)[1]
                pixels_info = re.findall(r'\[\d+,\d+]', line)[2]
                size = [int(n) for n in size_info.strip('[]').split(',')]
                windows = [int(n) for n in windows_info.strip('[]').split(',')]
                pixels = [int(n) for n in pixels_info.strip('[]').split(',')]

                stats["sizes"].append({
                    "size": size,
                    "windows": windows,
                    "pixels": pixels
                })
            # extract picture information part I
            elif line.startswith(" Picture"):
                parts = line.split(":")
                picture_name = parts[0].split()[-1]  # Extract picture name

                # Use regular expressions to find numbers
                numbers = re.findall(r'\d+', parts[1])
                if len(numbers) == 2:
                    homol_files, raw_homol_couples = map(int, numbers)  # Convert strings to integers
                    stats["pictures"].append({
                        "name": picture_name,
                        "homol_files": homol_files,
                        "raw_homol_couples": raw_homol_couples,
                    })
            # extract picture information part II
            elif line.startswith(" -"):
                # Extract coverage and points information
                parts = line.split(":")
                picture_name = parts[0].strip()[2:].strip()
                coverage, points = parts[1].split("(")
                # Clean up and convert the coverage percentage to a float
                coverage_percentage = float(coverage.strip().split('%')[0])
                # Extract and convert the points covered to an integer
                points_covered = int(points.split()[0])
                # Find the corresponding picture and update its information
                for pic in stats["pictures"]:
                    if pic["name"] == picture_name:
                        pic.update({
                            "coverage_percentage": coverage_percentage,
                            "points_covered": points_covered,
                        })
                        break

        # Serialize the dictionary to a JSON string
        json_output = json.dumps(stats, indent=4)

        # save json_output to a file
        with open(f"{self.project_folder}/stats/schnaps_stats.json", "w") as file:
            file.write(json_output)

    def validate_mm_parameters(self):

        if "/" in self.mm_args["ImagePattern"]:
            raise ValueError("ImagePattern cannot contain '/'. Use a pattern like '*.tif' instead.")

    def validate_required_files(self):

        # check all tif files in images-subfolder and copy them to the project folder if not already there
        homol_files = glob.glob(self.project_folder + "/images/*.tif")
        for file in homol_files:
            base_name = os.path.basename(file)

            if os.path.isfile(self.project_folder + "/" + base_name) is False:
                shutil.copy(file, self.project_folder + "/" + base_name)
