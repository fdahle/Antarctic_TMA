import os
import glob
import shutil

from src.sfm.mm_commands._base_command import BaseCommand

class GCPBascule(BaseCommand):

    required_args = ["ImagePattern", "InputOrientation", "OutputOrientation",
                     "FileGroundControlPoints", "FileImageMeasurements"]
    allowed_args = ["ImagePattern", "InputOrientation", "OutputOrientation",
                    "FileGroundControlPoints", "FileImageMeasurements", "L1", "CPI",
                    "ShowU", "ShowD", "PatNLD", "NLDDegX", "NLDDegY", "NLDDegZ", "NLFR",
                    "NLShow"]

    def __init__(self, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the input parameters
        self.validate_mm_parameters()

    def build_shell_string(self):

        # build the basic shell command
        shell_string = f'GCPBascule {self.mm_args["ImagePattern"]} ' \
                       f'{self.mm_args["InputOrientation"]} {self.mm_args["OutputOrientation"]} ' \
                       f'{self.mm_args["FileGroundControlPoints"]} ' \
                       f'{self.mm_args["FileImageMeasurements"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        return shell_string

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